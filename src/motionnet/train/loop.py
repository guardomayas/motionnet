# src/motionnet/train/loop.py
"""Training loop.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from motionnet.dataset import batches
from motionnet.models import MotionCNN
from motionnet.models.cnn import unit
from motionnet.train.losses import laplacian_penalty


def build_model(cfg, info, device):
    m = cfg["model"]
    return MotionCNN(fps=info.fps, **m).to(device)


def compute_v_std(train, valid, device):
    """Per-component velocity std over the valid frames. Targets are divided by this."""
    v = train["vel_deg_s"][:, valid].reshape(-1, 2)    
    return v.std(0).to(device)


def _ckpt(path, model, cfg, tag, v_std, valid, info, history, **extra):
    """Atomic save: write .tmp then rename, so a killed run leaves no half file."""
    tmp = path.with_suffix(".tmp")
    torch.save(dict(cfg=cfg, tag=tag, state_dict=model.state_dict(),
                    v_std=v_std.cpu(), valid_start=valid.start,
                    gray_frames=info.gray_frames, fps=info.fps,
                    max_delay=model.max_delay, history=history, **extra), tmp)
    tmp.replace(path)


def run_epoch(model, loader, cfg, v_std, valid, device,
              optimizer=None, ramp=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    # loss_fn = loss_fn or nn.MSELoss()
    L = cfg["loss"]

    keys = ("total", "mse", "svar", "rate1", "rate2", "zE")
    sums = {k: torch.zeros((), device=device) for k in keys}
    err = torch.zeros(2, device=device)
    sq = torch.zeros(2, device=device)
    n = 0

    for batch in loader:
        movie = batch["movie"].to(device)
        target = batch["vel_deg_s"].to(device) / v_std

        pred = model(movie)
        p, t = pred[:, valid], target[:, valid]
        # mse = loss_fn(p, t)
        mse  = (p - t).pow(2).sum(-1).mean() #to match vector notation
        svar = model._local_v[:, valid].var(2, unbiased=False).sum(-1).mean()
        # svar = model._local_v[:, valid].var(2, unbiased=False).mean()
        
        total = mse + L["gamma_var"] * svar
        if L["lambda_sp"]:
            total = total + L["lambda_sp"] * laplacian_penalty(
                unit(model.spatial_kernel, (1, 2, 3)))
        total = total + ramp * (L["lambda_1"] * model._subunit_rate
                        + L["lambda_2"] * model.layer2_rate
                        + L.get("lambda_z", 0.0) * model._z_energy)

        if is_train:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["optim"]["grad_clip"])
            optimizer.step()

        B = movie.shape[0]
        n += B
        for k, val in zip(keys, (total, mse, svar, model._subunit_rate,
                           model.layer2_rate, model._z_energy)):
            sums[k] += val.detach() * B
        err += (p - t).detach().pow(2).sum((0, 1))
        sq += t.pow(2).sum((0, 1))

    out = {k: (s / n).item() for k, s in sums.items()}
    r2 = (1 - err / sq).cpu().numpy()      # targets are zero-mean by construction
    out["r2_x"], out["r2_y"] = float(r2[0]), float(r2[1])
    out["nmse"] = (err.sum() / sq.sum()).item()     # 1 = predict zero
    return out


def fit(cfg, data, device, ckpt_dir,
        tag=None, log_every=20, progress=True):
    """Train one model. Returns (model, history, ckpt_path).

    train_mem / val_mem / info come from motionnet.dataset.prepare_data. They are
    passed in rather than built here so a sweep pays the data cost once.
    """
    O, L = cfg["optim"], cfg["loss"]
    info = data.info
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"lam2{L['lambda_2']}_lamv{L.get('gamma_var', 0.0)}_seed{O['seed']}"

    torch.manual_seed(O["seed"])
    np.random.seed(O["seed"])

    model = build_model(cfg, info, device)
    valid = slice(info.gray_frames + model.max_delay - 1, None)
    v_std = compute_v_std(data.train, valid, device)
    g = torch.Generator().manual_seed(O["seed"])

    print("Velocity std",v_std)
    
    # train_dl = batches(data.train, O["batch_size"], shuffle=True, drop_last=True, generator=g)
    # val_dl = batches(data.val, O["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=O["lr"],
                                  weight_decay=O["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=O["epochs"], eta_min=O["eta_min"])

    history = {"train": [], "val": [], "lr": []}
    save = lambda path, **extra: _ckpt(path, model, cfg, tag, v_std, valid,
                                       info, history, **extra)
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
        train_dl = batches(data.train, O["batch_size"], shuffle=True, drop_last=True, generator=g)
        val_dl = batches(data.val, O["batch_size"])
        tr = run_epoch(model, train_dl, cfg, v_std, valid, device, optimizer, ramp)
        with torch.no_grad():
            va = run_epoch(model, val_dl, cfg, v_std, valid, device, None, ramp)

        history["lr"].append(optimizer.param_groups[0]["lr"])
        sched.step()
        history["train"].append(tr)
        history["val"].append(va)

        if epoch % 50 == 0:      # periodic snapshot, survives a Colab disconnect
            save(ckpt_dir / f"{tag}_last.pt", epoch=epoch,
                 optimizer=optimizer.state_dict(), sched=sched.state_dict())
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
    save(ckpt_path, wall_s=time.time() - t0)
    (ckpt_dir / f"{tag}_cfg.json").write_text(json.dumps(cfg, indent=2, default=str))

    (ckpt_dir / f"{tag}_last.pt").unlink(missing_ok=True)   # <-- NEW: snapshot no longer needed

    return model, history, ckpt_path


def load_checkpoint(path, device, info=None):
    """Rebuild a model from a checkpoint. info is optional — fps is stored in the file."""
    ck = torch.load(path, map_location=device, weights_only=False)
    fps = info.fps if info is not None else ck["fps"]
    model = MotionCNN(fps=fps, **ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck