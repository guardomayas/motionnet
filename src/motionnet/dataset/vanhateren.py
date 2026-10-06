# vanhateren.py: corpus I/O and optics, pure functions
# load_iml(path) -> np.ndarray                         # (1024, 1536) float32, raw luminance
# split_images(corpus, n_images, val_frac, seed, stride) -> (list[Path], list[Path])
# prefilter(img, sigma_px, log=True, zscore=True) -> np.ndarray   # spline coeffs, float32


"""Van Hateren corpus: I/O, train/val split, optical prefilter."""

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, spline_filter

# spline_filter here and map_coordinates in stimulus.py must share these,
# or the B-spline prefilter isn't inverted near the border.
BOUNDARY_MODE = "reflect"
SPLINE_ORDER = 3
BLUR_TRUNCATE = 3.0          # sigmas; also sets the gaze window's border margin
FWHM_TO_SIGMA = 2.3548

VH_SHAPE = (1024, 1536)
IML_BYTES = VH_SHAPE[0] * VH_SHAPE[1] * 2      # big-endian uint16


def load_iml(path, dtype=np.float32):
    """Raw linear-luminance image, as stored."""
    return np.fromfile(path, dtype=">u2").reshape(VH_SHAPE).astype(dtype)

def valid_iml_files(data_path):
    """List .iml files matching VH_SHAPE, skipping corrupt/truncated ones.

    Van Hateren downloads occasionally include a partial or wrong-size file
    (an interrupted copy, a different acquisition format); reshaping one of
    those crashes make_movies deep into a run. Checking the raw byte
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

def split_images(data_path, n_images=200, val_frac=0.2, seed=0, stride=1, block=10):
    """Split .iml files into disjoint train/val lists.

    Files are numbered by acquisition, so neighbours are often the same scene
    minutes apart. Contiguous blocks are assigned whole, so near-duplicates
    can't straddle the split; `stride` additionally thins the corpus.
    """
    files = valid_iml_files(data_path)[::stride][:n_images]
    if not files:
        raise FileNotFoundError(f"no valid .iml files under {data_path}")
    blocks = [files[i:i + block] for i in range(0, len(files), block)]
    perm = np.random.default_rng(seed).permutation(len(blocks))
    n_val = max(1, round(val_frac * len(blocks)))
    val = [f for b in perm[:n_val] for f in blocks[b]]
    train = [f for b in perm[n_val:] for f in blocks[b]]
    return train, val

def sigma_for(blur_px, rho_phi_ratio=1.08):
    """Acceptance-function sigma (source px) from Δρ as an FWHM."""
    return blur_px * rho_phi_ratio / FWHM_TO_SIGMA


def normalize(img, log=True, zscore=True):
    """Raw luminance -> log, z-scored. Shared by prefilter and plotting."""
    if img.min() < 0:
        raise ValueError("expects raw non-negative luminance; "
                         "prefiltering twice widens Δρ by √2 silently")
    if log:
        img = np.log1p(img)
    if zscore:
        img = (img - img.mean()) / (img.std() + 1e-8)
    return img


def prefilter(img, sigma_px, log=True, zscore=True):
    """Raw image -> blurred B-spline coefficients, float32."""
    img = normalize(np.asarray(img, np.float32), log, zscore)
    if sigma_px > 0:
        img = gaussian_filter(img, sigma_px, mode=BOUNDARY_MODE, truncate=BLUR_TRUNCATE)
    return spline_filter(img, order=SPLINE_ORDER, output=np.float32, mode=BOUNDARY_MODE)