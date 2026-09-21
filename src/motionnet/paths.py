"""Environment-dependent paths. Importable from notebooks and cluster jobs."""

import shutil
from pathlib import Path
import yaml

DRIVE = Path("/content/drive/MyDrive/NUIN")


def setup(in_colab=False, n_images=None):
    """Resolve DATA_PATH and CKPT_DIR, copying images to local disk on Colab.

    Drive is high-latency per file, so reading .iml files straight off the
    mount costs minutes at dataset construction. Copy once to /content.
    """
    if not in_colab:
        data = Path("~/NUIN/van_hateren/vanhateren_iml").expanduser()
        ckpt = Path("runs")
    else:
        data = Path("/content/vanhateren")
        if not data.exists():
            data.mkdir(parents=True)
            src = sorted((DRIVE / "van_hateren").glob("*.iml"))[:n_images]
            for f in src:
                shutil.copy(f, data)
        ckpt = DRIVE / "motionnet_runs"

    ckpt.mkdir(parents=True, exist_ok=True)
    return data, ckpt

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