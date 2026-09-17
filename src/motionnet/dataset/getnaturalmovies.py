"""Translating van Hateren crops rendered onto a fly receptor lattice."""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset

from .vanhateren_utils import VH_SHAPE, load_iml
from .prefilter import (BLUR_TRUNCATE, BOUNDARY_MODE, SPLINE_ORDER,
                        PrefilterConfig, check_cache, sigma_for)
 
# Note bounds_mode is unrelated to prefilter.BOUNDARY_MODE
class GetNaturalMovies(Dataset):
    """Translating van Hateren crops rendered onto a fly receptor lattice.

    Angular scale is *declared*, not measured: one interommatidial angle
    (delta_phi_deg) is defined to span `blur_px` source pixels, following
    the Flyvis convention. The van Hateren nominal 1/60 deg/px is ignored.

    Boundary handling (`bounds_mode`):
      "reject"  -- redraw until the trace stays in bounds. Velocity is an
                   exact OU sample; the kept set is p(v | in bounds), a
                   describable truncation of the tails. Use for analysis.
      "reflect" -- fold the position trace at the walls. Always one draw, but
                   velocity reverses discontinuously at the boundary, so p(v)
                   is distorted in a position-dependent way. Use while
                   iterating on the pipeline.

    Convention: pos/vel column 0 is x (columns), column 1 is y (rows).
    """

    def __init__(
        self,
        data_path=None,
        files=None,                 # explicit list; overrides globbing
        n_images=None,              # take the first N when globbing
        log_image=True,
        normalize_images=True,
        coeff_cache=None,           # path to a build_coeff_cache() .npy
        # --- eye geometry ---
        blur_px=13,                 # acceptance width in source px; fixes Δρ
        spacing_px=None,            # receptor lattice; None -> = blur_px
        eye_size=28,
        delta_phi_deg=5.3,          # interommatidial angle
        rho_phi_ratio=1.08,         # acceptance / spacing
        blur=True,
        # --- timing ---
        frames_per_segment=75,
        fps=75,
        gray_sec=0.2,
        gray_value=0.0,             # pairs with z-scored images
        # --- motion ---
        samples_per_image=6,
        vel_half_life_s=0.2,
        vel_std_deg_s=(100.0, 100.0),
        vel_corr=0.0,               # correlation between vx and vy
        start_jitter=0.0,
        # --- boundary handling ---
        bounds_mode="reject",       # "reject" | "reflect"
        max_tries=20,               # redraws before falling back to reflect
        accept_warn_frac=0.8,       # warn below this acceptance rate
        bounds_warn_after=50,       # ...once this many traces are drawn
        # --- contrast ---
        contrast_range=None,        # e.g. (0.1, 1.0)
        noise_std=0.0,
        seed=20260818,
        movie_format="TCHW",
        dtype=torch.float32,
    ):
        if bounds_mode not in ("reject", "reflect"):
            raise ValueError("bounds_mode must be 'reject' or 'reflect'")

        self.data_path = data_path
        if files is not None:
            self.files = [Path(f) for f in files]
        else:
            if data_path is None:
                raise ValueError("pass either `files` or `data_path`")
            self.files = sorted(Path(data_path).glob("*.iml"))[:n_images]
        if not self.files:
            raise FileNotFoundError("no .iml files found")

        self.n_imgs = len(self.files)
        self.names = [f.name for f in self.files]
        # Stable identity for seeding: a hash of the filename, not the
        # position in this split, so a sample's trace is unchanged by
        # train/val slicing. (A previous version concatenated digit
        # characters from the stem, which silently collides for stems that
        # mix digits from more than one field, e.g. "imk01_v2" vs "imk012".)
        self._file_ids = [
            int(hashlib.sha256(f.stem.encode()).hexdigest()[:15], 16)
            for f in self.files
        ]
        self.H, self.W = VH_SHAPE
        self.eye_size = int(eye_size)
        self.blur_px = float(blur_px)
        self.spacing_px = float(blur_px if spacing_px is None else spacing_px)
        self.oversample = self.blur_px / self.spacing_px  # 1.0 = one/ommatidium
        self.delta_phi_deg = float(delta_phi_deg)
        self.dpp = self.delta_phi_deg / self.blur_px      # deg/px set by optics
        self.deg_per_tap = self.delta_phi_deg / self.oversample
        self.blur = blur
        
        # Everything that determines the coefficients, in one object: it is
        # what gets written to a cache's sidecar and what gets compared on
        # load, so the build and the check cannot disagree.
        self.cfg = PrefilterConfig(
            sigma_px=sigma_for(blur_px, rho_phi_ratio, blur),
            log_image=log_image,
            normalize_images=normalize_images,
        )
        
        # Acceptance function width (source px), Δρ as an FWHM -> sigma.
        self.sigma_px = self.cfg.sigma_px
        self.log_image = self.cfg.log_image
        self.normalize_images = self.cfg.normalize_images
        
        self.T = int(frames_per_segment)
        self.fps = float(fps)
        self.gray_frames = int(round(gray_sec * fps))
        self.gray_value = float(gray_value)

        self.samples_per_image = int(samples_per_image)
        self.tau = vel_half_life_s / np.log(2)
        vs = ((float(vel_std_deg_s),) * 2 if np.isscalar(vel_std_deg_s)
              else tuple(map(float, vel_std_deg_s)))
        self.vel_std_deg_s = vs
        self.vel_std = tuple(v / self.dpp for v in vs)    # -> source px/s
        self.vel_corr = float(vel_corr)
        self.start_jitter = float(start_jitter)

        self.bounds_mode = bounds_mode
        self.max_tries = int(max_tries)
        self.accept_warn_frac = float(accept_warn_frac)
        self.bounds_warn_after = int(bounds_warn_after)

        self.contrast_range = contrast_range
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.movie_format = movie_format
        self.dtype = dtype

        # Excursion budget: half the eye span, the start jitter, and a margin
        # for the blur's mirrored tails at the image border.
        self.half_px = (self.eye_size - 1) / 2 * self.spacing_px
        edge = BLUR_TRUNCATE * self.sigma_px
        self.lim_y = self.H / 2 - self.half_px - self.start_jitter - edge
        self.lim_x = self.W / 2 - self.half_px - self.start_jitter - edge
        if min(self.lim_x, self.lim_y) <= 0:
            raise ValueError(
                f"eye spans {2 * self.half_px:.0f} px, too large for "
                f"{VH_SHAPE} with jitter {self.start_jitter} and blur margin "
                f"{edge:.0f}")
        self._check_excursion_budget()

        self._coeffs = self._load_coeffs(coeff_cache)

        off = (np.arange(self.eye_size)
               - (self.eye_size - 1) / 2) * self.spacing_px
        self._gy, self._gx = np.meshgrid(off, off, indexing="ij")

        # Draw tallies (per worker process).
        self._n_samples = 0     # traces returned
        self._n_draws = 0       # OU draws made, including rejected ones
        self._n_fallback = 0    # samples that exhausted max_tries
        self._warned = False

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def _load_coeffs(self, coeff_cache):
        """Blurred spline coefficients, from cache if it matches this config."""
        if coeff_cache is not None:
            path = Path(coeff_cache)
            if path.exists():
                return check_cache(path, self.files, self.cfg)
            warnings.warn(f"{path} not found; prefiltering in memory instead. "
                          f"Run build_coeff_cache() to create it.",
                          RuntimeWarning, stacklevel=3)
 
        print(f"Prefiltering {len(self.files)} images.")
        return np.stack([self.cfg.apply(load_iml(f)) for f in self.files])

    def source_image(self, i, dtype=np.float32):
        """Raw (log, z-scored) image i -- for plotting, not rendering.
 
        Re-reads from disk each call; never use this inside __getitem__.
        Shares `cfg` with the rendering path, so it cannot drift out of step
        with the movies.
        """
        return self.cfg.source_image(self.files[i], dtype=dtype)


    # ------------------------------------------------------------------
    # Bounds diagnostics
    # ------------------------------------------------------------------
    def _pos_std_px(self):
        """Analytic std of OU position at t = T/fps, per axis (source px)."""
        t = self.T / self.fps
        f = np.sqrt(2 * (t / self.tau - 1 + np.exp(-t / self.tau)))
        return tuple(s * self.tau * f for s in self.vel_std)

    def _check_excursion_budget(self):
        """Warn up front if the excursion is likely to exceed the image."""
        sx, sy = self._pos_std_px()
        mode = ("many redraws" if self.bounds_mode == "reject"
                else "frequent reflections")
        for axis, s, lim in (("x", sx, self.lim_x), ("y", sy, self.lim_y)):
            if s > 0 and lim / s < 3.0:
                warnings.warn(
                    f"excursion budget is tight on {axis}: limit {lim:.0f} px "
                    f"is {lim / s:.1f} sigma of the position spread "
                    f"({s:.0f} px). In '{self.bounds_mode}' mode this means "
                    f"{mode}. Reduce vel_std_deg_s (now {self.vel_std_deg_s}), "
                    f"frames_per_segment (now {self.T}), or start_jitter "
                    f"(now {self.start_jitter:.0f}).",
                    RuntimeWarning, stacklevel=3)

    @property
    def acceptance_rate(self):
        """Fraction of OU draws accepted so far (this process)."""
        return self._n_samples / self._n_draws if self._n_draws else float("nan")

    def _tally(self, tries, fell_back):
        self._n_samples += 1
        self._n_draws += tries
        self._n_fallback += int(fell_back)
        if self._warned or self._n_samples < self.bounds_warn_after:
            return
        rate = self.acceptance_rate
        if rate < self.accept_warn_frac:
            self._warned = True
            extra = (f" {self._n_fallback} sample(s) exhausted max_tries and "
                     f"fell back to reflection." if self._n_fallback else "")
            warnings.warn(
                f"acceptance rate {100 * rate:.0f}% over {self._n_samples} "
                f"samples ({self._n_draws} draws): traces are leaving the "
                f"scene bounds often, so p(v) is truncated well inside its "
                f"tails and sampling is {1 / rate:.1f}x slower than needed."
                f"{extra} Reduce vel_std_deg_s (now {self.vel_std_deg_s}), "
                f"frames_per_segment (now {self.T}), or start_jitter "
                f"(now {self.start_jitter:.0f}).",
                RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------
    @staticmethod
    def _reflect(x, lim):
        """Fold x into [-lim, lim] by mirroring (triangle wave, period 4*lim)."""
        p = 4.0 * lim
        y = np.mod(x + lim, p)
        y = np.where(y > 2 * lim, p - y, y)
        return y - lim

    def _draw_ou(self, rng):
        """One stationary 2D OU velocity trace and its integrated position."""
        dt = 1.0 / self.fps
        alpha = np.exp(-dt / self.tau)
        sx, sy = self.vel_std
        r = self.vel_corr
        cov = np.array([[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]])

        noise = rng.multivariate_normal(np.zeros(2), cov, size=self.T)
        vel = np.empty((self.T, 2))
        vel[0] = rng.multivariate_normal(np.zeros(2), cov)  # stationary start
        b = np.sqrt(1 - alpha ** 2)
        for t in range(1, self.T):
            vel[t] = alpha * vel[t - 1] + b * noise[t]

        pos = np.cumsum(vel, axis=0) * dt
        pos -= pos[0]
        return vel, pos

    def _in_bounds(self, pos):
        return bool((np.abs(pos[:, 0]) <= self.lim_x).all()
                    and (np.abs(pos[:, 1]) <= self.lim_y).all())

    def _apply_reflection(self, pos):
        """Fold position, then recompute velocity so labels match pixels."""
        pos = pos.copy()
        pos[:, 0] = self._reflect(pos[:, 0], self.lim_x)
        pos[:, 1] = self._reflect(pos[:, 1], self.lim_y)
        vel = np.zeros_like(pos)
        vel[1:] = np.diff(pos, axis=0) * self.fps
        vel[0] = vel[1]
        return vel, pos

    def generate_velocity_trace_2d(self, rng):
        """Return (vel, pos, tries, fell_back) honouring `bounds_mode`."""
        if not -1.0 <= self.vel_corr <= 1.0:
            raise ValueError("vel_corr must be in [-1, 1]")

        if self.bounds_mode == "reflect":
            vel, pos = self._draw_ou(rng)
            if not self._in_bounds(pos):
                vel, pos = self._apply_reflection(pos)
            return vel.astype(np.float32), pos.astype(np.float32), 1, False

        for tries in range(1, self.max_tries + 1):
            vel, pos = self._draw_ou(rng)
            if self._in_bounds(pos):
                return (vel.astype(np.float32), pos.astype(np.float32),
                        tries, False)

        # Exhausted: fall back to reflection rather than looping forever.
        vel, pos = self._apply_reflection(pos)
        return (vel.astype(np.float32), pos.astype(np.float32),
                self.max_tries, True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_movie(self, img_idx, cx, cy, pos):
        coeffs = self._coeffs[img_idx]
        dx = pos[:, 0, None, None]
        dy = pos[:, 1, None, None]
        yy = self._gy[None] + cy + dy    # (T, n, n)
        xx = self._gx[None] + cx + dx
        return map_coordinates(coeffs, [yy, xx], order=SPLINE_ORDER,
                               mode=BOUNDARY_MODE,
                               prefilter=False).astype(np.float32)

    def add_gray_padding(self, movie):
        gray = np.full((self.gray_frames, *movie.shape[1:]),
                       self.gray_value, dtype=movie.dtype)
        return np.concatenate([gray, movie], axis=0)

    def pad_position_trace(self, pos):
        return np.concatenate([np.repeat(pos[:1], self.gray_frames, 0), pos])

    def pad_velocity_trace(self, vel):
        pre = np.zeros((self.gray_frames, vel.shape[1]), dtype=vel.dtype)
        return np.concatenate([pre, vel])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return self.n_imgs * self.samples_per_image

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(
                f"index {idx} out of range for {len(self)} samples")
        img_idx, rep = divmod(idx, self.samples_per_image)
        file_id = self._file_ids[img_idx]
        rng = np.random.default_rng((self.seed, file_id, rep))

        vel, pos, tries, fell_back = self.generate_velocity_trace_2d(rng)
        self._tally(tries, fell_back)

        j = self.start_jitter
        cx = self.W / 2 + rng.uniform(-j, j)
        cy = self.H / 2 + rng.uniform(-j, j)

        movie = self.render_movie(img_idx, cx, cy, pos)

        # Remove local luminance: the gaze window's mean depends on where it
        # landed (sky vs foliage), which is a nuisance variable here. Done
        # unconditionally so the input distribution doesn't depend on whether
        # contrast is being manipulated.
        movie = movie - movie.mean()

        # Contrast drawn independently of velocity, applied as gain on the
        # zero-mean movie; fixed-variance noise is what makes gain informative.
        gain = 1.0
        if self.contrast_range is not None:
            gain = float(rng.uniform(*self.contrast_range))
            movie = movie * gain
        rms = float(movie.std())        # after gain, before noise
        if self.noise_std > 0:
            movie = movie + rng.normal(
                0, self.noise_std, movie.shape).astype(np.float32)

        movie = self.add_gray_padding(movie)

        pos = self.pad_position_trace(pos)
        vel = self.pad_velocity_trace(vel)

        if self.movie_format == "TCHW":
            movie = movie[:, None, :, :]
        elif self.movie_format == "THW":
            pass
        else:
            raise ValueError(f"Unknown movie_format: {self.movie_format}")

        return {
            "movie": torch.from_numpy(movie).to(self.dtype),
            "pos_px": torch.from_numpy(pos).to(self.dtype),
            "vel_px_s": torch.from_numpy(vel).to(self.dtype),
            "vel_deg_s": torch.from_numpy(vel * self.dpp).to(self.dtype),
            "vel_omm_frame": torch.from_numpy(
                vel / (self.blur_px * self.fps)).to(self.dtype),
            "vel_tap_frame": torch.from_numpy(
                vel / (self.spacing_px * self.fps)).to(self.dtype),
            "contrast_gain": torch.tensor(gain, dtype=self.dtype),
            "contrast_rms": torch.tensor(rms, dtype=self.dtype),
            "snr": torch.tensor(rms / self.noise_std if self.noise_std > 0 #amplitude ratio snr, some other Lit uses poser (snr^2)
                                else float("inf"), dtype=self.dtype),
            "gaze_center": torch.tensor([cx, cy], dtype=self.dtype),
            "n_tries": torch.tensor(tries),
            "bounds_fallback": torch.tensor(fell_back),
            "image_idx": torch.tensor(file_id),
        }