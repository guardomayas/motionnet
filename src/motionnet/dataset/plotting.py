import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


### ------------------------------------
### ------------------------------------
### plotting utils
### ------------------------------------
### ------------------------------------


def show_hdr(img, ax=None, mode="log", lo=1, hi=99.5, title=None,
             extent=None, horizon_row=None):
    """Display-only tone map. Input: raw or scaled linear radiance."""
    x = np.asarray(img, dtype=np.float64)
    pos = x[x > 0]
    floor = np.percentile(pos, 0.1) if pos.size else 1e-8

    if mode == "log":
        y = np.log10(np.maximum(x, floor))
    elif mode == "gamma":
        y = np.power(np.maximum(x, 0) / np.percentile(x, hi), 1 / 2.2)
    else:
        y = x

    vmin, vmax = np.percentile(y, [lo, hi])
    ax = ax or plt.gca()
    ax.imshow(y, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal", extent=extent)
    if horizon_row is not None:
        ax.axhline(horizon_row, color="tab:red", lw=0.5)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    return ax

def plot_examples(imgs, n_rows, n_cols, random_seed=0, **kwargs):
    n_samples = n_rows * n_cols
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(imgs), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for ax, i in zip(axes.flat[:n_samples], idx):
        show_hdr(imgs[i, :, :], ax=ax, **kwargs)

    for ax in axes.flat[n_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def animate_sample(d, idx=0, stride=1, interval=None):
    s = d[idx]
    movie  = s["movie"][:, 0].numpy()
    pos    = s["pos_px"].numpy()            # (T, 2): col 0 = x, col 1 = y
    vel    = s["vel_deg_s"].numpy()
    cx, cy = s["gaze_center"].numpy()
    img    = d.source_image(idx// d.samples_per_image)

    if interval is None:
        interval = 1000 * stride / d.fps    # ms per displayed frame

    span     = 2 * d.half_px
    t_s      = np.arange(len(vel)) / d.fps
    half_deg = (d.eye_size - 1) / 2 * d.delta_phi_deg + d.delta_phi_deg / 2
    vmin, vmax = np.percentile(movie, [1, 99])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    axs[0].imshow(img, cmap="gray")
    rect = plt.Rectangle((cx - d.half_px, cy - d.half_px), span, span,
                         ec="r", fc="none", lw=1.5)
    axs[0].add_patch(rect)
    axs[0].set_title("source image + gaze window")

    im = axs[1].imshow(movie[0], cmap="gray", vmin=vmin, vmax=vmax,
                       interpolation="nearest",
                       extent=[-half_deg, half_deg, -half_deg, half_deg])
    axs[1].set_xlabel("azimuth (deg)")
    axs[1].set_ylabel("elevation (deg)")
    axs[1].set_title(f"{d.eye_size}x{d.eye_size} receptors")

    axs[2].plot(t_s, vel[:, 0], label="vx", lw=1)
    axs[2].plot(t_s, vel[:, 1], label="vy", lw=1)
    axs[2].axhline(0, color="k", lw=0.5)
    cursor = axs[2].axvline(0, color="r", lw=1.5)
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("velocity (deg/s)")
    axs[2].legend(loc="upper right", fontsize=8)
    # flag = " — BOUNDS HIT" if s["bounds_hit"] else ""
    axs[2].set_title(f"velocity")
    at_lim = ((np.abs(pos[:, 0]) > 0.999 * d.lim_x) |
              (np.abs(pos[:, 1]) > 0.999 * d.lim_y))
    for t in np.flatnonzero(at_lim):
        axs[2].axvline(t_s[t], color="k", lw=2, alpha=0.5)

    def update(t):
        im.set_data(movie[t])
        rect.set_xy((cx + pos[t, 0] - d.half_px,
                     cy + pos[t, 1] - d.half_px))
        cursor.set_xdata([t_s[t], t_s[t]])
        return im, rect, cursor

    anim = FuncAnimation(fig, update, frames=range(0, len(movie), stride),
                         interval=interval, blit=False)
    plt.close()
    return anim
