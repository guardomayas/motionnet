"""Van Hateren corpus: file discovery, raw loading, train/val splits."""

import warnings
from pathlib import Path

import numpy as np

VH_SHAPE = (1024, 1536)
IML_BYTES = VH_SHAPE[0] * VH_SHAPE[1] * 2      # big-endian uint16


def load_iml(path, dtype=np.float32):
    """Raw linear-luminance image, as stored."""
    return np.fromfile(path, dtype=">u2").reshape(VH_SHAPE).astype(dtype)


def valid_iml_files(data_path):
    """List .iml files matching VH_SHAPE, skipping corrupt/truncated ones.

    Van Hateren downloads occasionally include a partial or wrong-size file
    (an interrupted copy, a different acquisition format); reshaping one of
    those crashes build_coeff_cache deep into a run. Checking the raw byte
    count up front is cheap and catches it before any work is wasted.
    """
    files = sorted(Path(data_path).glob("*.iml"))
    good = [f for f in files if f.stat().st_size == IML_BYTES]
    bad = [f for f in files if f.stat().st_size != IML_BYTES]
    if bad:
        names = ", ".join(f.name for f in bad[:5])
        warnings.warn(
            f"skipping {len(bad)} .iml file(s) with unexpected size under "
            f"{data_path} (expected {IML_BYTES} bytes for {VH_SHAPE}): "
            f"{names}{', ...' if len(bad) > 5 else ''}"
        )
    return good


def split_images(data_path, n_images=200, val_frac=0.2, seed=0, stride=1):
    """Split .iml files into disjoint train/val lists.

    Van Hateren images are numbered by acquisition, so consecutive files are
    often the same location minutes apart. `stride` thins the corpus; the
    permutation then keeps near-duplicates from straddling the split.
    """
    files = valid_iml_files(data_path)[::stride][:n_images]
    if not files:
        raise FileNotFoundError(f"no valid .iml files under {data_path}")
    perm = np.random.default_rng(seed).permutation(len(files))
    n_val = int(round(val_frac * len(files)))
    val = [files[i] for i in perm[:n_val]]
    train = [files[i] for i in perm[n_val:]]
    return train, val


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


# def scale_luminance(img, percentile_scale=99, clip=True, eps=1e-8):
#     """
#     Scale image for training after filtering.
#     """
#     img = img.astype(np.float32)

#     if percentile_scale is not None:
#         scale = np.percentile(img, percentile_scale)
#         img = img / (scale + eps)

#     if clip:
#         img = np.clip(img, 0, 1)

#     return img.astype(np.float32)

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

