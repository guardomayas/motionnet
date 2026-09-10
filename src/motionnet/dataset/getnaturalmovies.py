import os
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter
from torch.utils.data import Dataset

VH_SHAPE = (1024, 1536)


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
        data_path,
        batch_idx=0,
        batch_size=5,
        log_image=True,
        normalize_images=True,
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
        self.batch_idx = int(batch_idx)
        self.batch_size = int(batch_size)
        self.log_image = log_image
        self.normalize_images = normalize_images

        self.eye_size = int(eye_size)
        self.blur_px = float(blur_px)
        self.spacing_px = float(blur_px if spacing_px is None else spacing_px)
        self.oversample = self.blur_px / self.spacing_px   # 1.0 = one/ommatidium
        self.delta_phi_deg = float(delta_phi_deg)
        self.dpp = self.delta_phi_deg / self.blur_px       # deg/px set by optics
        self.deg_per_tap = self.delta_phi_deg / self.oversample
        self.blur = blur

        self.T = int(frames_per_segment)
        self.fps = float(fps)
        self.gray_frames = int(round(gray_sec * fps))
        self.gray_value = float(gray_value)

        self.samples_per_image = int(samples_per_image)
        self.tau = vel_half_life_s / np.log(2)
        vs = ((float(vel_std_deg_s),) * 2 if np.isscalar(vel_std_deg_s)
              else tuple(map(float, vel_std_deg_s)))
        self.vel_std_deg_s = vs
        self.vel_std = tuple(v / self.dpp for v in vs)     # -> source px/s
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

        self.batch_imgs, self.names = self._load_batch()
        self.n_imgs, self.H, self.W = self.batch_imgs.shape

        # Acceptance function width (source px).
        self.sigma_px = (self.blur_px * float(rho_phi_ratio) / 2.3548
                         if blur else 0.0)

        # Excursion budget: half the eye span, the start jitter, and a margin
        # for the blur's mirrored tails at the image border.
        self.half_px = (self.eye_size - 1) / 2 * self.spacing_px
        edge = 3.0 * self.sigma_px
        self.lim_y = self.H / 2 - self.half_px - self.start_jitter - edge
        self.lim_x = self.W / 2 - self.half_px - self.start_jitter - edge
        if min(self.lim_x, self.lim_y) <= 0:
            raise ValueError(
                f"eye spans {2 * self.half_px:.0f} px, too large for "
                f"{VH_SHAPE} with jitter {self.start_jitter} and blur margin "
                f"{edge:.0f}")

        self._check_excursion_budget()

        self._coeffs = np.stack([self._prep(im) for im in self.batch_imgs])

        off = (np.arange(self.eye_size)
               - (self.eye_size - 1) / 2) * self.spacing_px
        self._gy, self._gx = np.meshgrid(off, off, indexing="ij")

        # Draw tallies (per worker process).
        self._n_samples = 0     # traces returned
        self._n_draws = 0       # OU draws made, including rejected ones
        self._n_fallback = 0    # samples that exhausted max_tries
        self._warned = False

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
        for axis, s, lim in (("x", sx, self.lim_x), ("y", sy, self.lim_y)):
            if s > 0 and lim / s < 3.0:
                warnings.warn(
                    f"excursion budget is tight on {axis}: limit {lim:.0f} px "
                    f"is {lim / s:.1f} sigma of the position spread "
                    f"({s:.0f} px). In '{self.bounds_mode}' mode this means "
                    f"{'many redraws' if self.bounds_mode == 'reject' else 'frequent reflections'}. "
                    f"Reduce vel_std_deg_s (now {self.vel_std_deg_s}), "
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
    # Images
    # ------------------------------------------------------------------
    def _load_batch(self, dtype=np.float32):
        path = Path(self.data_path)
        iml_files = sorted(path.glob("*.iml"))
        start = self.batch_idx * self.batch_size
        batch_files = iml_files[start:start + self.batch_size]
        if not batch_files:
            raise FileNotFoundError(f"no .iml files for batch {self.batch_idx}")
        print(f"Found {len(iml_files)} .iml in {path}; loading batch "
              f"{self.batch_idx} ({len(batch_files)} images).")

        imgs, names = [], []
        for fname in batch_files:
            raw = np.fromfile(fname, dtype=">u2").reshape(VH_SHAPE)
            names.append(os.path.basename(fname))
            img = raw.astype(dtype)
            if self.log_image:
                img = np.log1p(img)                  # safe at raw == 0
            if self.normalize_images:
                img = (img - img.mean()) / (img.std() + 1e-8)
            imgs.append(img)
        return np.stack(imgs), names

    def _prep(self, im):
        """Blur by the acceptance function, then take spline coefficients."""
        if self.sigma_px > 0:
            im = gaussian_filter(im, sigma=self.sigma_px,
                                 mode="reflect", truncate=3.0)
        return spline_filter(im, order=3, output=np.float32)

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
        vel[0] = rng.multivariate_normal(np.zeros(2), cov)   # stationary start
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
                return vel.astype(np.float32), pos.astype(np.float32), tries, False

        # Exhausted: fall back to reflection rather than looping forever.
        vel, pos = self._apply_reflection(pos)
        return vel.astype(np.float32), pos.astype(np.float32), self.max_tries, True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_movie(self, img_idx, cx, cy, pos):
        n = self.eye_size
        out = np.empty((self.T, n, n), dtype=np.float32)
        coeffs = self._coeffs[img_idx]
        for t in range(self.T):
            dx, dy = pos[t]
            out[t] = map_coordinates(
                coeffs,
                [self._gy + cy + dy, self._gx + cx + dx],
                order=3, mode="reflect", prefilter=False,
            )
        return out

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
        img_idx, rep = divmod(idx, self.samples_per_image)
        global_idx = self.batch_idx * self.batch_size + img_idx
        rng = np.random.default_rng((self.seed, global_idx, rep))

        vel, pos, tries, fell_back = self.generate_velocity_trace_2d(rng)
        self._tally(tries, fell_back)

        j = self.start_jitter
        cx = self.W / 2 + rng.uniform(-j, j)
        cy = self.H / 2 + rng.uniform(-j, j)

        movie = self.render_movie(img_idx, cx, cy, pos)

        # Contrast drawn independently of velocity, applied as gain on the
        # zero-mean movie; fixed-variance noise is what makes gain informative.
        gain = 1.0
        if self.contrast_range is not None:
            gain = float(rng.uniform(*self.contrast_range))
            movie = movie * gain
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
            "contrast": torch.tensor(gain, dtype=self.dtype),
            "gaze_center": torch.tensor([cx, cy], dtype=self.dtype),
            "n_tries": torch.tensor(tries),
            "bounds_fallback": torch.tensor(fell_back),
            "image_idx": global_idx,
            "name": self.names[img_idx],
        }