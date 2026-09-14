"""Environment-dependent paths. Importable from notebooks and cluster jobs."""

import shutil
from pathlib import Path

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