# src/motionnet/train/loop.py
"""Training loop extracted from notebooks/fit_CNN.ipynb (cells 12-13).

Behaviour is intentionally identical to the notebook: same penalties, same
renormalization after each step, same warmup ramp, same VALID slice. The only
additions are (a) no module-level globals and (b) checkpoints get written.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from motionnet.models import MotionCNN
from motionnet.train.losses import laplacian_penalty

# dims must match what forward() normalizes over
SCALE_FREE = {
    "spatial_kernel": (1, 2, 3),    # per layer-1 channel
    "temporal_w":     (1,),         # per row; phi = w @ B.T is normalized downstream
    "layer2.weight":  (1, 2, 3),    # per layer-2 channel
    "readout":        (1,),         # per output row
}


def normed(w, dims):
    return w / (torch.linalg.vector_norm(w, dim=dims, keepdim=True) + 1e-8)


@torch.no_grad()
def renormalize(model):
    params = dict(model.named_parameters())
    for name, dims in SCALE_FREE.items():
        p = params[name]
        p.div_(torch.linalg.vector_norm(p, dim=dims, keepdim=True) + 1e-8)


def build_model(cfg, info, device):
    m = cfg["model"]
    return MotionCNN(fps=info.fps, **m).to(device)


def compute_v_std(train_mem, valid, device):
    """Per-component velocity std over the valid frames. Targets are divided by this."""
    v = train_mem.tensors["vel_deg_s"][:, valid].reshape(-1, 2)
    return v.std(0).to(device)


def run_epoch(model, loader, cfg, v_std, valid, device,
              optimizer=None, ramp=1.0, loss_fn=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = loss_fn or nn.MSELoss()
    L = cfg["loss"]

    keys = ("total", "mse", "rate1", "rate2")
    sums = {k: torch.zeros((), device=device) for k in keys}
    err = torch.zeros(2, device=device)
    sq = torch.zeros(2, device=device)
    n = 0

    for batch in loader:
        movie = batch["movie"].to(device)
        target = batch["vel_deg_s"].to(device) / v_std

        pred = model(movie)
        p, t = pred[:, valid], target[:, valid]
        mse = loss_fn(p, t)

        total = mse
        if L["lambda_sp"]:
            total = total + L["lambda_sp"] * laplacian_penalty(
                normed(model.spatial_kernel, (1, 2, 3)))
        total = total + ramp * (L["lambda_1"] * model._subunit_rate
                                + L["lambda_2"] * model.layer2_rate)

        if is_train:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["optim"]["grad_clip"])
            optimizer.step()
            renormalize(model)

        B = movie.shape[0]
        n += B
        for k, val in zip(keys, (total, mse, model._subunit_rate, model.layer2_rate)):
            sums[k] += val.detach() * B
        err += (p - t).detach().pow(2).sum((0, 1))
        sq += t.pow(2).sum((0, 1))

    out = {k: (s / n).item() for k, s in sums.items()}
    r2 = (1 - err / sq).cpu().numpy()      # targets are zero-mean by construction
    out["r2_x"], out["r2_y"] = float(r2[0]), float(r2[1])
    return out


def fit(cfg, train_mem, val_mem, info, device, ckpt_dir,
        tag=None, log_every=20, progress=True):
    """Train one model. Returns (model, history, ckpt_path).

    train_mem / val_mem / info come from motionnet.dataset.prepare_data. They are
    passed in rather than built here so a sweep pays the data cost once.
    """
    O, L = cfg["optim"], cfg["loss"]
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"lam2{L['lambda_2']}_seed{O['seed']}"

    torch.manual_seed(O["seed"])
    np.random.seed(O["seed"])

    model = build_model(cfg, info, device)
    valid = slice(info.gray_frames + model.max_delay - 1, None)
    v_std = compute_v_std(train_mem, valid, device)
    print("Velocity std",v_std)
    train_dl = DataLoader(train_mem, batch_size=O["batch_size"], shuffle=True,
                          drop_last=True, num_workers=0)
    val_dl = DataLoader(val_mem, batch_size=O["batch_size"], num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=O["lr"],
                                  weight_decay=O["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=O["epochs"], eta_min=O["eta_min"])

    history = {"train": [], "val": [], "lr": []}
    t0 = time.time()
    epochs = range(1, O["epochs"] + 1)
    if progress:
        try:
            from tqdm import tqdm
            epochs = tqdm(epochs, desc=tag)
        except ImportError:
            pass

    for epoch in epochs:
        ramp = min(1.0, epoch / L["warmup_epochs"])
        tr = run_epoch(model, train_dl, cfg, v_std, valid, device, optimizer, ramp)
        with torch.no_grad():
            va = run_epoch(model, val_dl, cfg, v_std, valid, device, None, ramp)

        history["lr"].append(optimizer.param_groups[0]["lr"])
        sched.step()
        history["train"].append(tr)
        history["val"].append(va)

        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == O["epochs"]):
            msg = (f"{tag} {epoch:3d} | mse {tr['mse']:.3f}/{va['mse']:.3f} | "
                   f"R2 val x {va['r2_x']:.3f} y {va['r2_y']:.3f} | "
                   f"rate1 {va['rate1']:.3f} rate2 {va['rate2']:.3f}")
            try:
                from tqdm import tqdm
                tqdm.write(msg)
            except ImportError:
                print(msg, flush=True)

    ckpt_path = ckpt_dir / f"{tag}.pt"
    torch.save({
        "cfg": cfg,
        "tag": tag,
        "state_dict": model.state_dict(),
        "v_std": v_std.cpu(),
        "valid_start": valid.start,
        "gray_frames": info.gray_frames,
        "fps": info.fps,
        "max_delay": model.max_delay,
        "history": history,
        "wall_s": time.time() - t0,
    }, ckpt_path)
    (ckpt_dir / f"{tag}_cfg.json").write_text(json.dumps(cfg, indent=2, default=str))

    return model, history, ckpt_path


def load_checkpoint(path, info, device):
    """Rebuild a model from a checkpoint. Returns (model, ckpt_dict)."""
    ck = torch.load(path, map_location=device)
    model = build_model(ck["cfg"], info, device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
