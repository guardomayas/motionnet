# src/motionnet/paths.py
"""Environment-dependent paths. Importable from notebooks and cluster jobs."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

# Persistent copy of the van Hateren .iml files on Drive. Adjust if it lives elsewhere.
COLAB_CORPUS_SRC = Path("/content/drive/MyDrive/NUIN/van_hateren/vanhateren_iml")


@dataclass(frozen=True)
class Paths:
    corpus: Path
    coeffs: Path
    movies: Path
    ckpt: Path


def _stage_corpus(src: Path, dst: Path) -> None:
    """Copy the corpus from Drive to local disk once per runtime (FUSE reads are slow)."""
    if any(dst.glob("*.iml")):
        return
    if not src.is_dir():
        raise FileNotFoundError(f"corpus source not found on Drive: {src}")
    print(f"staging corpus {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def setup(in_colab: bool) -> Paths:
    if in_colab:
        if not os.path.ismount("/content/drive"):
            raise RuntimeError("Drive not mounted")
        root = Path("/content/drive/MyDrive/NUIN/motionnet")
        p = Paths(corpus=Path("/content/vanhateren"),       # scratch: local copy of Drive corpus
                  coeffs=root / "cache",                    # persistent: Drive
                  movies=Path("/content/movie_cache"),      # scratch: local disk
                  ckpt=root / "ckpt")
        _stage_corpus(COLAB_CORPUS_SRC, p.corpus)
    else:
        root = Path.home() / "NUIN/motionnet/data"
        p = Paths(corpus=Path.home() / "NUIN/van_hateren/vanhateren_iml",
                  coeffs=root / "coeffs",
                  movies=root / "coeffs",
                  ckpt=root / "ckpt")
    for d in (p.coeffs, p.movies, p.ckpt):
        d.mkdir(parents=True, exist_ok=True)
    return p


REPO = Path(__file__).resolve().parents[2]     # src/motionnet/paths.py -> repo root
CONFIGS = REPO / "configs"


def load_cfg(kind, name):
    """load_cfg('stimulus', 'eye31_150fps') -> dict"""
    return yaml.safe_load((CONFIGS / kind / f"{name}.yaml").read_text())


def compose(exp_name=None, data=None, model=None, corpus=None, **overrides):
    """Build one resolved config. Either name an experiment or a data+model pair."""
    spec = load_cfg("exp", exp_name) if exp_name else {}
    data = data or spec["data"]                    # exp YAMLs use `data:`
    model = model or spec["model"]

    cfg = load_cfg("model", model)
    cfg["stimulus"] = load_cfg("stimulus", data)
    cfg["stimulus"]["corpus"] = str(corpus)        # environment, not config
    cfg["meta"] = dict(sweep=spec.get("name", "adhoc"),
                       data_id=data, model_id=model)

    for k, v in overrides.items():                 # "loss.lambda_2" -> 0.04
        *path, last = k.split(".")
        dd = cfg
        for q in path:
            dd = dd[q]
        dd[last] = v
    return cfg, spec.get("grid", {})