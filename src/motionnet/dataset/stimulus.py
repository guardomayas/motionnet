"""
Natural-motion stimuli: translating van Hateren crops on a fly receptor lattice.
    make_movies(files, params, seed) -> (arrays, stats)   the physics; pure
    get_data(stim_cfg, cache_dir, device) -> Data         cache + device
    batches(split, batch_size, ...) -> Iterator[dict]     minibatches
"""
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from tqdm.auto import tqdm

from .vanhateren import (BLUR_TRUNCATE, BOUNDARY_MODE, SPLINE_ORDER, VH_SHAPE,
                         load_iml, prefilter, sigma_for, split_images)

CACHE_VERSION = 1   # bump when make_movies' output changes for the same params


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MovieParams:
    """Everything that determines a movie. Unknown YAML keys raise TypeError.

    Angular scale is declared, not measured: one interommatidial angle
    (delta_phi_deg) spans `blur_px` source px (Flyvis convention).
    Velocity column 0 is x (image columns), column 1 is y (rows).

    bounds_mode:
      "reject"  -- redraw until the trace stays in bounds; kept velocities are
                   p(v | in bounds). Falls back to reflection after max_tries.
      "reflect" -- fold position at the walls; one draw, but p(v) is distorted.
    """
    # eye geometry
    eye_size: int = 28
    blur_px: float = 13.0              # acceptance FWHM in source px; fixes Δρ
    spacing_px: float | None = None    # receptor lattice; None -> blur_px
    delta_phi_deg: float = 5.3
    rho_phi_ratio: float = 1.08
    log_image: bool = True
    zscore: bool = True
    # timing
    frames_per_segment: int = 75
    fps: float = 75.0
    gray_sec: float = 0.2
    gray_value: float = 0.0            # pairs with z-scored images
    # motion
    samples_per_image: int = 6
    vel_half_life_s: float = 0.2
    vel_std_deg_s: tuple = (100.0, 100.0)
    vel_corr: float = 0.0
    start_jitter: float = 0.0
    bounds_mode: str = "reject"
    max_tries: int = 20
    # contrast and noise
    contrast_range: tuple | None = None
    noise_std: float = 0.0

    def __post_init__(self):
        v = self.vel_std_deg_s
        object.__setattr__(self, "vel_std_deg_s",
                           (float(v),) * 2 if np.isscalar(v) else tuple(map(float, v)))
        if self.contrast_range is not None:
            object.__setattr__(self, "contrast_range",
                               tuple(map(float, self.contrast_range)))
        if self.bounds_mode not in ("reject", "reflect"):
            raise ValueError("bounds_mode must be 'reject' or 'reflect'")
        if not -1.0 <= self.vel_corr <= 1.0:
            raise ValueError("vel_corr must be in [-1, 1]")
        if min(self.lim) <= 0:
            raise ValueError(f"eye spans {2 * self.half_px:.0f} px: too large for "
                             f"{VH_SHAPE} with jitter {self.start_jitter} and "
                             f"blur margin {BLUR_TRUNCATE * self.sigma_px:.0f}")
        self._warn_if_tight()

    # --- derived quantities ---
    @property
    def spacing(self):      return float(self.blur_px if self.spacing_px is None else self.spacing_px)
    @property
    def dpp(self):          return self.delta_phi_deg / self.blur_px      # deg per source px
    @property
    def deg_per_tap(self):  return self.dpp * self.spacing
    @property
    def sigma_px(self):     return sigma_for(self.blur_px, self.rho_phi_ratio)
    @property
    def T(self):            return int(self.frames_per_segment)
    @property
    def gray_frames(self):  return int(round(self.gray_sec * self.fps))
    @property
    def tau(self):          return self.vel_half_life_s / math.log(2)
    @property
    def vel_std_px(self):   return tuple(s / self.dpp for s in self.vel_std_deg_s)
    @property
    def half_px(self):      return (self.eye_size - 1) / 2 * self.spacing

    @property
    def lim(self):
        """Max |excursion| (x, y) in source px: half the image, minus eye,
        start jitter, and the blur's mirrored border."""
        H, W = VH_SHAPE
        m = self.half_px + self.start_jitter + BLUR_TRUNCATE * self.sigma_px
        return W / 2 - m, H / 2 - m

    def _warn_if_tight(self):
        t = self.T / self.fps
        f = math.sqrt(2 * (t / self.tau - 1 + math.exp(-t / self.tau)))
        for axis, s, lim in zip("xy", self.vel_std_px, self.lim):
            pos_std = s * self.tau * f        # analytic OU position std at t
            if pos_std > 0 and lim / pos_std < 3.0:
                warnings.warn(
                    f"excursion budget tight on {axis}: limit {lim:.0f} px is "
                    f"{lim / pos_std:.1f} sigma of position spread. Expect "
                    f"{'many redraws' if self.bounds_mode == 'reject' else 'frequent reflections'}; "
                    f"reduce vel_std_deg_s, frames_per_segment or start_jitter.",
                    RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Velocity traces
# ---------------------------------------------------------------------------
def _draw_ou(p, rng):
    """Stationary 2D OU velocity (source px/s) and integrated position."""
    dt = 1.0 / p.fps
    alpha = np.exp(-dt / p.tau)
    sx, sy = p.vel_std_px
    r = p.vel_corr
    cov = np.array([[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]])

    noise = rng.multivariate_normal(np.zeros(2), cov, size=p.T)
    vel = np.empty((p.T, 2))
    vel[0] = rng.multivariate_normal(np.zeros(2), cov)      # stationary start
    b = np.sqrt(1 - alpha ** 2)
    for t in range(1, p.T):
        vel[t] = alpha * vel[t - 1] + b * noise[t]
    pos = np.cumsum(vel, axis=0) * dt
    return vel, pos - pos[0]


def _fold(pos, p):
    """Mirror position into bounds (triangle wave), recompute velocity."""
    def tri(x, lim):
        per = 4.0 * lim
        y = np.mod(x + lim, per)
        return np.where(y > 2 * lim, per - y, y) - lim
    lx, ly = p.lim
    pos = np.stack([tri(pos[:, 0], lx), tri(pos[:, 1], ly)], axis=1)
    vel = np.zeros_like(pos)
    vel[1:] = np.diff(pos, axis=0) * p.fps
    vel[0] = vel[1]
    return vel, pos


def _trace(p, rng):
    """(vel, pos, tries, fell_back), float32, honouring bounds_mode."""
    lx, ly = p.lim
    n_max = 1 if p.bounds_mode == "reflect" else p.max_tries
    for tries in range(1, n_max + 1):
        vel, pos = _draw_ou(p, rng)
        if (np.abs(pos[:, 0]) <= lx).all() and (np.abs(pos[:, 1]) <= ly).all():
            return vel.astype(np.float32), pos.astype(np.float32), tries, False
    vel, pos = _fold(pos, p)
    return (vel.astype(np.float32), pos.astype(np.float32),
            tries, p.bounds_mode == "reject")


def _file_id(path):
    """Seed identity from the file stem, so a sample's trace doesn't depend
    on where the file landed in a split."""
    return int(hashlib.sha256(Path(path).stem.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def make_movies(files, p, seed, progress=True):
    """Render samples_per_image movies per file.

    Returns (arrays, stats). arrays share leading axis N = len(files) * S:
      movie         (N, G+T, 1, E, E) float16   gray padding first
      vel_deg_s     (N, G+T, 2)       float32   zero during gray
      contrast_gain (N,)              float32
      contrast_rms  (N,)              float32   after gain, before noise
      image_id      (N,)              int64     _file_id of the source image
    """
    files = [Path(f) for f in files]
    S, G, T, E = p.samples_per_image, p.gray_frames, p.T, p.eye_size
    N = len(files) * S
    out = dict(movie=np.empty((N, G + T, 1, E, E), np.float16),
               vel_deg_s=np.zeros((N, G + T, 2), np.float32),
               contrast_gain=np.empty(N, np.float32),
               contrast_rms=np.empty(N, np.float32),
               image_id=np.empty(N, np.int64))
    out["movie"][:, :G] = p.gray_value

    off = (np.arange(E) - (E - 1) / 2) * p.spacing
    gy, gx = np.meshgrid(off, off, indexing="ij")
    H, W = VH_SHAPE
    j = p.start_jitter
    draws = fallbacks = 0

    i = 0
    for f in tqdm(files, desc="movies", disable=not progress):
        coeffs = prefilter(load_iml(f), p.sigma_px, p.log_image, p.zscore)
        fid = _file_id(f)
        for rep in range(S):
            rng = np.random.default_rng((seed, fid, rep))
            vel, pos, tries, fell = _trace(p, rng)
            draws += tries
            fallbacks += fell

            cx = W / 2 + rng.uniform(-j, j)
            cy = H / 2 + rng.uniform(-j, j)
            yy = gy[None] + cy + pos[:, 1, None, None]
            xx = gx[None] + cx + pos[:, 0, None, None]
            movie = map_coordinates(coeffs, [yy, xx], order=SPLINE_ORDER,
                                    mode=BOUNDARY_MODE,
                                    prefilter=False).astype(np.float32)

            # Gaze-window mean (sky vs foliage) is a nuisance variable.
            movie = movie - movie.mean()
            gain = 1.0
            if p.contrast_range is not None:
                gain = float(rng.uniform(*p.contrast_range))
                movie = movie * gain
            rms = float(movie.std())
            if p.noise_std > 0:
                movie = movie + rng.normal(0, p.noise_std, movie.shape).astype(np.float32)

            out["movie"][i, G:, 0] = movie
            out["vel_deg_s"][i, G:] = vel * p.dpp
            out["contrast_gain"][i] = gain
            out["contrast_rms"][i] = rms
            out["image_id"][i] = fid
            i += 1

    stats = dict(acceptance=N / draws if draws else float("nan"),
                 fallbacks=int(fallbacks))
    if stats["acceptance"] < 0.8:
        warnings.warn(f"acceptance {stats['acceptance']:.0%}: p(v) is truncated "
                      f"well inside its tails", RuntimeWarning, stacklevel=2)
    return out, stats


# ---------------------------------------------------------------------------
# Cache and device
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataInfo:
    fps: float
    T: int
    gray_frames: int
    dpp: float
    deg_per_tap: float
    sigma_px: float
    spacing_px: float
    n_train: int
    n_val: int
    train_acceptance: float
    val_acceptance: float
    train_fallbacks: int
    val_fallbacks: int
    key: str


@dataclass(frozen=True)
class Data:
    train: dict = field(repr=False)     # str -> Tensor, on device
    val: dict = field(repr=False)
    info: DataInfo


def get_data(stim_cfg, cache_dir, device="cpu", progress=True):
    """Config -> device-resident train/val tensors, generating at most once.

    The cache filename is a hash of everything that determines the output.
    The corpus *path* is excluded (it differs between Colab and local), the
    file *names* are included. Writes go to .tmp then rename, so a killed run
    never leaves a file under a real name.
    """
    p = MovieParams(**stim_cfg["movie"])
    train_f, val_f = split_images(stim_cfg["corpus"], **stim_cfg["split"])
    seeds = (stim_cfg["ds_seed_train"], stim_cfg["ds_seed_val"])
    spec = dict(version=CACHE_VERSION, movie=asdict(p), seeds=seeds,
                train=[f.name for f in train_f], val=[f.name for f in val_f])
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"stim_{key}.pt"

    if not path.exists():
        blob = {}
        for name, files, seed in (("train", train_f, seeds[0]),
                                  ("val", val_f, seeds[1])):
            arrays, stats = make_movies(files, p, seed, progress)
            blob[name] = {k: torch.from_numpy(v) for k, v in arrays.items()}
            blob[f"{name}_stats"] = stats
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(blob, tmp)
        tmp.replace(path)
        path.with_suffix(".json").write_text(json.dumps(spec, indent=1))  # for humans

    blob = torch.load(path, weights_only=True)
    info = DataInfo(
        fps=p.fps, T=p.T, gray_frames=p.gray_frames, dpp=p.dpp,
        deg_per_tap=p.deg_per_tap, sigma_px=p.sigma_px, spacing_px=p.spacing,
        n_train=len(blob["train"]["movie"]), n_val=len(blob["val"]["movie"]),
        train_acceptance=blob["train_stats"]["acceptance"],
        val_acceptance=blob["val_stats"]["acceptance"],
        train_fallbacks=blob["train_stats"]["fallbacks"],
        val_fallbacks=blob["val_stats"]["fallbacks"], key=key)
    to = lambda d: {k: v.to(device) for k, v in d.items()}
    return Data(train=to(blob["train"]), val=to(blob["val"]), info=info)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def batches(split, batch_size, shuffle=False, drop_last=False, generator=None):
    """Yield index-sliced minibatches from a dict of equal-length tensors.

    float16 tensors come out as float32. `generator` is a CPU torch.Generator,
    so shuffle order is reproducible without touching global RNG state.
    """
    first = next(iter(split.values()))
    n, dev = len(first), first.device
    idx = (torch.randperm(n, generator=generator) if shuffle
           else torch.arange(n)).to(dev)
    end = n - n % batch_size if drop_last else n
    for a in range(0, end, batch_size):
        j = idx[a:a + batch_size]
        yield {k: (v[j].float() if v.dtype == torch.float16 else v[j])
               for k, v in split.items()}