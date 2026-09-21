"""The prefilter chain: log -> z-score -> blur -> spline coefficients.

Coefficients are expensive to compute and are indexed positionally, so a cache
that doesn't match the config it's loaded against produces silently wrong
movies rather than an error. Everything that determines the coefficients lives
in `PrefilterConfig`, which is what gets written to the sidecar and what gets
compared on load -- so the writer and the reader cannot disagree about which
parameters matter.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter, spline_filter

from .vanhateren_utils import VH_SHAPE, load_iml

# `spline_filter` and `map_coordinates` must agree on the boundary mode, or the
# prefilter is not inverted near the image border. Unrelated to the dataset's
# `bounds_mode` argument, which folds velocity traces.
BOUNDARY_MODE = "reflect"
SPLINE_ORDER = 3
BLUR_TRUNCATE = 3.0     # gaussian_filter truncation, in sigmas; also sets the
                        # border margin the gaze window has to stay inside
CACHE_VERSION = 1       # bump to invalidate every cache at once

FWHM_TO_SIGMA = 2.3548

def movie_fingerprint(seed, CFG, PRE, n):
    return json.dumps({"cfg": CFG, "pre": PRE.__dict__, "seed": seed, "n": n},
                      sort_keys=True, default=str)

def sigma_for(blur_px, rho_phi_ratio=1.08, blur=True):
    """Acceptance-function width in source px, from an FWHM (Delta rho)."""
    return float(blur_px) * float(rho_phi_ratio) / FWHM_TO_SIGMA if blur else 0.0


@dataclass(frozen=True)
class PrefilterConfig:
    """Everything that determines the coefficients. Serialized to the sidecar.

    Adding a field here automatically makes it recorded at build time and
    checked at load time; that's the point of the dataclass. Bump
    CACHE_VERSION when you add one.
    """
    sigma_px: float
    log_image: bool = True
    normalize_images: bool = True
    boundary_mode: str = BOUNDARY_MODE
    spline_order: int = SPLINE_ORDER
    blur_truncate: float = BLUR_TRUNCATE

    # Compared with np.isclose rather than ==; the rest must match exactly.
    _FLOAT_FIELDS = ("sigma_px", "blur_truncate")

    def apply(self, img):
        """Run the full chain on one raw image -> float32 spline coefficients."""
        img = np.asarray(img, dtype=np.float32)
        if img.min() < 0:
            raise ValueError( #only deteces on normalized images
                "apply() expects raw non-negative luminance; got data with "
                f"min {img.min():.3g}. Applying the prefilter chain twice "
                "widens Delta_rho by sqrt(2) and is otherwise silent.")
        if self.log_image:
            img = np.log1p(img)                       # safe at raw == 0
        if self.normalize_images:
            img = (img - img.mean()) / (img.std() + 1e-8)
        if self.sigma_px > 0:
            img = gaussian_filter(img, sigma=self.sigma_px,
                                  mode=self.boundary_mode,
                                  truncate=self.blur_truncate)
        return spline_filter(img, order=self.spline_order, output=np.float32,
                             mode=self.boundary_mode)

    def source_image(self, path, dtype=np.float32):
        """Raw image with log/z-score applied but no blur -- for plotting.

        Shares the branches above so a plotted source image can't drift out of
        step with the movie rendered from the same file.
        """
        img = load_iml(path, dtype=dtype)
        if self.log_image:
            img = np.log1p(img)
        if self.normalize_images:
            img = (img - img.mean()) / (img.std() + 1e-8)
        return img


def _metadata(cfg, files):
    return {"cache_version": CACHE_VERSION,
            "files": [str(f) for f in files],
            **asdict(cfg)}


def build_coeff_cache(files, out_path, cfg):
    """Prefilter each image once; store as a float32 .npy with a sidecar .json.

    The sidecar is written last, on purpose: see check_cache.
    """
    out_path = Path(out_path)
    files = [Path(f) for f in files]
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(len(files), *VH_SHAPE))
    for i, f in enumerate(tqdm(files, desc=out_path.name)):
        arr[i] = cfg.apply(load_iml(f))
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")
    arr.flush()

    out_path.with_suffix(".json").write_text(json.dumps(_metadata(cfg, files)))
    return out_path.parent.mkdir(parents=True, exist_ok=True)


def check_cache(path, files, cfg):
    """Validate a coefficient cache against `cfg` and `files`.

    Coefficients are indexed positionally and carry the whole prefilter chain,
    so every parameter that touched them has to match or the movies are
    silently wrong.

    Files are compared by name, not by full path: the same corpus lives under
    different directories locally and on Colab, and a check that can't survive
    that just gets switched off by the next person in a hurry.

    Returns the memmapped array, so the caller doesn't reopen it.
    """
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        # build_coeff_cache allocates the .npy first and writes the sidecar
        # last, so "array, no json" is the signature of a run killed partway
        # through: full-size file, zeros past the point it died. Training on
        # that is worse than not training at all.
        raise ValueError(
            f"{path} has no sidecar .json, so nothing about it can be "
            f"verified. It may be a partial cache from an interrupted "
            f"build_coeff_cache() run, in which case its tail is zeros. "
            f"Rebuild it.")

    meta = json.loads(meta_path.read_text())

    # Version first: on a bump the other keys may be absent or may have
    # changed meaning, so there is nothing useful to say about them.
    if meta.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"{path} has cache_version "
                         f"{meta.get('cache_version')!r}, this code writes "
                         f"{CACHE_VERSION}; rebuild it")

    if [Path(f).name for f in files] != [Path(p).name
                                         for p in meta.get("files", [])]:
        raise ValueError(f"{path} was built from a different file list "
                         f"(or a different order); rebuild it")

    for key, want in asdict(cfg).items():
        got = meta.get(key)
        if key in PrefilterConfig._FLOAT_FIELDS:
            if key == "blur_truncate" and cfg.sigma_px == 0:
                continue                  # truncation is moot without blur
            ok = got is not None and np.isclose(got, want)
        else:
            ok = got == want
        if not ok:
            raise ValueError(f"{path} was built with {key}={got!r}, this "
                             f"dataset wants {want!r}; rebuild it")

    # Structural check, independent of whether the metadata is honest about
    # what produced the array.
    arr = np.load(path, mmap_mode="r")
    want_shape = (len(files), *VH_SHAPE)
    if arr.shape != want_shape or arr.dtype != np.float32:
        raise ValueError(f"{path} has shape {arr.shape} dtype {arr.dtype}; "
                         f"expected {want_shape} float32")
    return arr


def ensure_cache(files, out_path, cfg):
    """Load the cache if it's valid, rebuild it if it isn't.

    Replaces the notebook's `if not path.exists()` guard, which skips the
    rebuild for a cache that exists but is stale -- and stale is exactly what
    every cache becomes on a CACHE_VERSION bump.
    """
    out_path = Path(out_path)
    if out_path.exists():
        try:
            return check_cache(out_path, files, cfg)
        except ValueError as e:
            print(f"rebuilding cache: {e}")
    build_coeff_cache(files, out_path, cfg)
    return check_cache(out_path, files, cfg)