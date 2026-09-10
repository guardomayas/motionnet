from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_vanhateren_raw(path, dtype=np.float32):
    """
    Load Van Hateren .iml or .imc binary image as raw linear luminance.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=">u2")  # big-endian uint16

    if raw.size == 1024 * 1536:
        img = raw.reshape((1024, 1536))

    elif raw.size == 768 * 1024:
        img = raw.reshape((768, 1024))

    else:
        raise ValueError(
            f"Unexpected size for {path.name}: {raw.size} uint16 values"
        )

    return img.astype(dtype)


def scale_luminance(img, percentile_scale=99, clip=True, eps=1e-8):
    """
    Scale image for training after filtering.
    """
    img = img.astype(np.float32)

    if percentile_scale is not None:
        scale = np.percentile(img, percentile_scale)
        img = img / (scale + eps)

    if clip:
        img = np.clip(img, 0, 1)

    return img.astype(np.float32)

def log_image(path, eps=1.0):
    """Log-luminance. Percentile scaling is redundant here: dividing by a
    scalar becomes an additive constant under log, and per-patch mean
    subtraction removes it anyway."""
    raw = load_vanhateren_raw(path)
    return np.log(raw + eps)


## Processing utils
## random crops

def random_crop(img, rng, crop_hw=(50, 50)):
    H, W = img.shape
    crop_h, crop_w = crop_hw
    y0 = rng.integers(0, H - crop_h + 1)
    x0 = rng.integers(0, W - crop_w + 1)
    return img[y0:y0+crop_h, x0:x0+crop_w].copy()

