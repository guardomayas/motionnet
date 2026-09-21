
# src/motionnet/paths.py
"""Environment-dependent paths. Importable from notebooks and cluster jobs."""

import shutil
from pathlib import Path
import yaml

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    corpus: Path
    coeffs: Path
    movies: Path
    ckpt: Path

def setup(in_colab: bool) -> Paths:
    if in_colab:
        if not os.path.ismount("/content/drive"):
            raise RuntimeError("Drive not mounted")
        root = Path("/content/drive/MyDrive/NUIN/motionnet")
        p = Paths(corpus=Path("/content/vanhateren"),       # whatever setup() does now
                  coeffs=root / "cache",                    # persistent: Drive
                  movies=Path("/content/movie_cache"),      # scratch: local disk
                  ckpt=root / "ckpt")
    else:
        root = Path.home() / "NUIN/motionnet/data"
        p = Paths(corpus=..., coeffs=root / "coeffs",
                  movies=root / "coeffs", ckpt=root / "ckpt")
    for d in (p.coeffs, p.movies, p.ckpt):
        d.mkdir(parents=True, exist_ok=True)
    return p

REPO = Path(__file__).resolve().parents[2]     # src/motionnet/paths.py -> repo root
CONFIGS = REPO / "configs"

def load_cfg(kind, name):
    """load_cfg('data', 'eye31_150fps') -> dict"""
    return yaml.safe_load((CONFIGS / kind / f"{name}.yaml").read_text())

def compose(exp_name=None, data=None, model=None, corpus=None, **overrides):
    """Build one resolved config. Either name an experiment or a data+model pair."""
    spec = load_cfg("exp", exp_name) if exp_name else {}
    data = data or spec["stimulus"]
    model = model or spec["model"]

    cfg = load_cfg("model", model)
    cfg["stimulus"] = load_cfg("stimulus", data)
    cfg["stimulus"]["corpus"] = str(corpus)            # environment, not config
    cfg["meta"] = dict(sweep=spec.get("name", "adhoc"),
                       data_id=data, model_id=model)

    for k, v in overrides.items():                 # "loss.lambda_2" -> 0.04
        *path, last = k.split(".")
        dd = cfg
        for q in path:
            dd = dd[q]
        dd[last] = v
    return cfg, spec.get("grid", {})