"""Per-checkpoint unit analysis: performance, layer-2 tuning, coverage, sparseness.

Convention-free: preferred directions and DSIs come from how each unit's activity
covaries with the true velocity on natural movies, so no assumption about the sign
of image motion vs eye motion is needed.

    from motionnet.analysis.units import analyze_sweep, plot_vs
    runs, units = analyze_sweep(D, data, "cuda")          # D = checkpoint folder
    plot_vs(runs, "loss.lambda_r2")
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import convolve2d

from motionnet.dataset import batches
from motionnet.models.cnn import unit
from motionnet.train.loop import load_checkpoint

CARDINAL = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect(model, ck, data, device, max_batches=None):
    """Run the validation set once. Returns per-frame arrays (valid frames only)."""
    valid = slice(ck["valid_start"], None)
    v_std = ck["v_std"].to(device)
    model.eval()
    pools, preds, tgts = [], [], []
    a1 = r2c = y2c = a2 = 0.0
    loc_err = loc_n = 0.0
    nb = 0
    for i, b in enumerate(batches(data.val, 32)):
        if max_batches and i >= max_batches:
            break
        mv = b["movie"].to(device)
        B = mv.shape[0]
        p = model(mv)[:, valid]
        t = b["vel_deg_s"].to(device)[:, valid] / v_std
        pools.append(model._pool.reshape(B, -1, model._pool.shape[-1])[:, valid].reshape(-1, model._pool.shape[-1]).cpu())
        preds.append(p.reshape(-1, 2).cpu())
        tgts.append(t.reshape(-1, 2).cpu())
        lv = model._local_v[:, valid]                                    # (B, T, N, 2)
        loc_err += (lv - t[:, :, None]).pow(2).sum().item()
        loc_n += lv[..., 0].numel()
        a1 = a1 + model._active1.cpu()
        a2 = a2 + model._active2.cpu()
        r2c = r2c + getattr(model, "_rate2_c", torch.full_like(model._active2, float("nan"))).cpu()
        y2c = y2c + getattr(model, "_y2sq_c", torch.full_like(model._active2, float("nan"))).cpu()
        nb += 1
    T = torch.cat(tgts)
    return dict(pool=torch.cat(pools).numpy(), pred=torch.cat(preds).numpy(), tgt=T.numpy(),
                active1=(a1 / nb).numpy(), active2=(a2 / nb).numpy(),
                rate2_c=(r2c / nb).numpy(), y2sq_c=(y2c / nb).numpy(),
                local_nmse=loc_err / loc_n / T.var(0).mean().item() / 2)


# ---------------------------------------------------------------------------
# Linear space-time RF (secondary DS measure)
# ---------------------------------------------------------------------------
def layer2_st_rf(model):
    """(C2, D, H, W) linearised RF per layer-2 unit; exact only where layer 1 is linear."""
    S = unit(model.spatial_kernel, (1, 2, 3))[:, 0].detach().cpu().numpy()
    phi = unit(model.temporal_w @ model.tbasis.T, (1,)).detach().cpu().numpy()
    g = model.gain.detach().cpu().numpy()
    K = unit(model.layer2.weight, (1, 2, 3)).detach().cpu().numpy()
    C2, C1 = K.shape[:2]
    sp = np.stack([[g[c] * convolve2d(K[m, c], S[c], mode="full") for c in range(C1)] for m in range(C2)])
    return np.einsum("mcyx,cd->mdyx", sp, phi)


def st_dsi(rf2d):
    """Quadrant-energy index of a (lag, space) filter: +-1 = one direction only, 0 = separable."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(rf2d))) ** 2
    w = np.fft.fftshift(np.fft.fftfreq(rf2d.shape[0]))[:, None]
    k = np.fft.fftshift(np.fft.fftfreq(rf2d.shape[1]))[None, :]
    pos, neg = F[(w * k) > 0].sum(), F[(w * k) < 0].sum()
    return float((pos - neg) / (pos + neg + 1e-12))


# ---------------------------------------------------------------------------
# Unit and run tables
# ---------------------------------------------------------------------------
def _nearest_cardinal(angle, tol=30.0):
    for name, a in CARDINAL.items():
        if abs((angle - a + 180) % 360 - 180) <= tol:
            return name
    return "oblique"


def unit_table(model, st, thresh_v=0.5, used_w=0.05, used_active=0.01):
    """One row per layer-2 unit."""
    R = unit(model.readout, (1,)).detach().cpu().numpy()                 # (2, C2)
    pool, tgt = st["pool"], st["tgt"]
    rf = layer2_st_rf(model)
    rows = []
    for m in range(pool.shape[1]):
        y = pool[:, m]
        # empirical tuning: regression of activity on velocity (convention-free)
        X = np.c_[tgt, np.ones(len(tgt))]
        beta = np.linalg.lstsq(X, y, rcond=None)[0][:2]
        ang = float(np.degrees(np.arctan2(beta[1], beta[0])))
        ax = int(abs(beta[1]) > abs(beta[0]))                           # 0 = x, 1 = y
        sgn = np.sign(beta[ax])
        pref = sgn * tgt[:, ax] > thresh_v
        null = -sgn * tgt[:, ax] > thresh_v
        yp, yn = y[pref].mean(), y[null].mean()
        mean_y, mean_y2 = st["rate2_c"][m], st["y2sq_c"][m]
        rows.append(dict(
            unit=m + 1,
            w=float((R[:, m] ** 2).sum()),
            readout_angle=float(np.degrees(np.arctan2(R[1, m], R[0, m]))),
            pref_angle=ang,
            pref_dir=_nearest_cardinal(ang),
            vdsi=float((yp - yn) / (abs(yp) + abs(yn) + 1e-12)),        # natural-movie DSI along preferred axis
            null_over_pref=float(yn / (yp + 1e-12)),
            active=float(st["active2"][m]),
            rate=float(mean_y),
            sparseness=float(1 - mean_y ** 2 / (mean_y2 + 1e-12)),       # Treves-Rolls (lifetime, per location)
            st_dsi=abs(st_dsi(rf[m].sum(1 + (1 - ax)))),                 # |linear-RF DSI| on the preferred axis
        ))
    df = pd.DataFrame(rows)
    df["used"] = (df.w > used_w) & (df.active > used_active)
    df["always_on"] = df.active > 0.95
    return df


def run_row(u, st, model):
    """One row per checkpoint: summaries over units + performance."""
    pred, tgt = st["pred"], st["tgt"]
    used = u[u.used]
    dirs = set(used.loc[used.pref_dir != "oblique", "pref_dir"])
    row = dict(
        n_used=int(len(used)),
        n_dead=int((~u.used).sum()),
        n_always_on=int(u.always_on.sum()),
        n_dirs=len(dirs),
        n_opponent_axes=int({"+x", "-x"} <= dirs) + int({"+y", "-y"} <= dirs),
        vdsi_mean=float(used.vdsi.mean()) if len(used) else np.nan,
        null_over_pref=float(used.null_over_pref.median()) if len(used) else np.nan,
        st_dsi_mean=float(used.st_dsi.mean()) if len(used) else np.nan,
        sparseness=float(used.sparseness.mean()) if len(used) else np.nan,
        active2=float(u.active.mean()),
        active1=float(st["active1"].mean()),
        gain1_mean=model.gain.mean().item(),
        local_nmse=st["local_nmse"],
        bias_x=model.readout_bias[0].item(), bias_y=model.readout_bias[1].item(),
    )
    for k, ax in enumerate("xy"):                                       # shrinkage per direction
        for s, lab in ((1, "+"), (-1, "-")):
            msk = s * tgt[:, k] > 0.5
            row[f"gain_{lab}{ax}"] = float((pred[msk, k] * tgt[msk, k]).sum() / (tgt[msk, k] ** 2).sum())
    return row


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def analyze_sweep(folder, data, device, max_batches=None):
    """Analyse every finished checkpoint in a sweep folder.

    Returns (runs, units): runs has one row per checkpoint with the full config
    flattened into dotted columns ('loss.lambda_r2', ...) and the summary.csv metrics;
    units has one row per (checkpoint, layer-2 unit).
    """
    folder = Path(folder)
    summary = pd.read_csv(folder / "summary.csv") if (folder / "summary.csv").exists() else None
    runs, units = [], []
    for f in sorted(folder.glob("*.pt")):
        if f.stem.endswith("_last"):
            continue
        model, ck = load_checkpoint(f, device, data.info)
        st = collect(model, ck, data, device, max_batches)
        u = unit_table(model, st)
        cfg = _flat({k: ck["cfg"][k] for k in ("model", "loss", "optim")})
        row = {"tag": ck["tag"], **cfg, **run_row(u, st, model)}
        runs.append(row)
        units.append(u.assign(tag=ck["tag"], **{k: cfg[k] for k in ("loss.lambda_r1", "loss.lambda_r2", "loss.gamma_var",
                                                                     "model.layer2_noise_std", "optim.seed") if k in cfg}))
    runs = pd.DataFrame(runs)
    if summary is not None:
        keep = [c for c in summary.columns if c == "tag" or c not in runs.columns]
        runs = runs.merge(summary[keep], on="tag", how="left")
    return runs, pd.concat(units, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
DEFAULT_PANELS = [("nmse", "nmse"), ("rate2", "layer-2 rate"), ("sparseness", "sparseness (TR)"),
                  ("vdsi_mean", "DSI (natural movies)"), ("null_over_pref", "null / pref"),
                  ("n_dirs", "directions covered"), ("n_opponent_axes", "opponent axes"),
                  ("st_dsi_mean", "|linear-RF DSI|")]


def plot_vs(runs, xcol, panels=DEFAULT_PANELS, ncols=4):
    """Each panel: seeds as dots, mean +- sd as a line, x as categories (works for lambda = 0)."""
    import matplotlib.pyplot as plt
    xs = sorted(runs[xcol].unique())
    pos = {x: i for i, x in enumerate(xs)}
    panels = [(c, l) for c, l in panels if c in runs.columns]
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for a, (col, lab) in zip(axs.flat, panels):
        x = runs[xcol].map(pos)
        a.scatter(x + np.random.uniform(-0.08, 0.08, len(x)), runs[col], s=18, alpha=0.4, lw=0)
        m = runs.groupby(xcol)[col].agg(["mean", "std"]).reindex(xs)
        a.errorbar(range(len(xs)), m["mean"], yerr=m["std"], marker="o", lw=2, capsize=0)
        a.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        a.set(xlabel=xcol, ylabel=lab)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for a in axs.flat[len(panels):]:
        a.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Mechanism: where does direction selectivity come from, filter or threshold?
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_drive(model, ck, data, device, max_batches=None):
    """Per-frame (valid frames), per-unit: output y, linear drive energy, drive rectified at 0."""
    valid = slice(ck["valid_start"], None)
    v_std = ck["v_std"].to(device)
    model.eval()
    out = {k: [] for k in ("y", "drive_sq", "drive_rect", "tgt")}
    sd = []
    for i, b in enumerate(batches(data.val, 32)):
        if max_batches and i >= max_batches:
            break
        mv = b["movie"].to(device)
        B = mv.shape[0]
        model(mv)
        C2 = model._pool.shape[-1]
        f = lambda a: a.reshape(B, -1, C2)[:, valid].reshape(-1, C2).cpu()
        out["y"].append(f(model._pool))
        out["drive_sq"].append(f(model._drive_sq))
        out["drive_rect"].append(f(model._drive_rect))
        out["tgt"].append((b["vel_deg_s"].to(device)[:, valid] / v_std).reshape(-1, 2).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def _dsi(r, pref, null):
    a, b = r[pref].mean(), r[null].mean()
    return float((a - b) / (abs(a) + abs(b) + 1e-12))


def threshold_mechanism(folder, data, device, thresh_v=0.5, max_batches=None):
    """One row per (checkpoint, used layer-2 unit).

    dsi_linear : DSI of the linear drive's energy  <drive^2>   -> what the filter alone gives
    dsi_rect0  : DSI of the drive rectified at 0   <[drive]+>  -> filter + ReLU, no learned threshold
    dsi_output : DSI of the actual output          <h[drive - theta]+>
    theta_rel  : threshold in units of the drive's s.d.  (raw theta is meaningless if the drive scale changes)

    Threshold mechanism predicts: dsi_linear ~ dsi_rect0 low and flat across lambda,
    dsi_output rising with lambda, together with theta_rel.
    """
    rows = []
    for f in sorted(Path(folder).glob("*.pt")):
        if f.stem.endswith("_last"):
            continue
        model, ck = load_checkpoint(f, device, data.info)
        if not hasattr(model, "_drive_sq"):
            model(next(iter(batches(data.val, 2)))["movie"].to(device))
            if not hasattr(model, "_drive_sq"):
                raise AttributeError("cnn.py is missing the _drive_sq/_drive_rect diagnostics")
        d = collect_drive(model, ck, data, device, max_batches)
        R = unit(model.readout, (1,)).detach().cpu().numpy()
        th = model.threshold2.detach().cpu().numpy()
        X = np.c_[d["tgt"], np.ones(len(d["tgt"]))]
        for m in range(d["y"].shape[1]):
            y = d["y"][:, m]
            w = float((R[:, m] ** 2).sum())
            if w < 0.05 or (y > 0).mean() < 0.01:
                continue                                                 # unused unit
            beta = np.linalg.lstsq(X, y, rcond=None)[0][:2]
            ax = int(abs(beta[1]) > abs(beta[0]))
            sgn = np.sign(beta[ax])
            pref = sgn * d["tgt"][:, ax] > thresh_v
            null = -sgn * d["tgt"][:, ax] > thresh_v
            drive_sd = float(np.sqrt(d["drive_sq"][:, m].mean()))
            mu = float(d["drive_mean"][:, m].mean())
            sd = float(np.sqrt(d["drive_sq"][:, m].mean() - mu**2))
            
            rows.append({
                "tag": ck["tag"],
                "loss.lambda_r2": ck["cfg"]["loss"]["lambda_r2"],
                "loss.lambda_r1": ck["cfg"]["loss"]["lambda_r1"],
                "optim.seed": ck["cfg"]["optim"]["seed"],
                "unit": m + 1,
                "dsi_linear": _dsi(d["drive_sq"][:, m], pref, null),
                "dsi_rect0": _dsi(d["drive_rect"][:, m], pref, null),
                "dsi_output": _dsi(y, pref, null),
                "theta": float(th[m]),
                "theta_rel": float(th[m] / (drive_sd + 1e-12)),
                "gain2": float(model.gain2[m].item()),
                "drive_mean_rel": mu / sd,                 # where the drive sits relative to 0
                "z_eff": (th[m] - mu) / sd,                # effective threshold, in s.d. above the drive mean
                "net_K": float(unit(model.layer2.weight, (1, 2, 3))[m].sum().item()),   # kernel DC
            })
    return pd.DataFrame(rows)


def plot_mechanism(mech, xcol="loss.lambda_r2"):
    """Left: three DSIs vs lambda (per-run mean over used units). Right: relative threshold."""
    import matplotlib.pyplot as plt
    per_run = mech.groupby(["tag", xcol]).mean(numeric_only=True).reset_index()
    xs = sorted(per_run[xcol].unique())
    pos = {x: i for i, x in enumerate(xs)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for col, lab in (("dsi_linear", "linear drive"), ("dsi_rect0", "drive, ReLU at 0"),
                     ("dsi_output", "output (learned threshold)")):
        m = per_run.groupby(xcol)[col].agg(["mean", "std"]).reindex(xs)
        a.errorbar(range(len(xs)), m["mean"], yerr=m["std"], marker="o", lw=2, capsize=0, label=lab)
    a.set(xlabel=xcol, ylabel="DSI (natural movies)")
    a.legend(frameon=False, fontsize=9)
    m = per_run.groupby(xcol)["theta_rel"].agg(["mean", "std"]).reindex(xs)
    b.scatter(per_run[xcol].map(pos), per_run["theta_rel"], s=18, alpha=0.4, lw=0)
    b.errorbar(range(len(xs)), m["mean"], yerr=m["std"], marker="o", lw=2, capsize=0, c="k")
    b.axhline(0, ls=":", c="k", lw=0.8)
    b.set(xlabel=xcol, ylabel=r"threshold / s.d.(drive)")
    for ax in (a, b):
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
