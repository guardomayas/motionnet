import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def temporal_kernels(cnn):
    """(N1, D) kernels in plot orientation: lag 0 LAST, matching t_ms."""
    phi = cnn.temporal_w @ cnn.tbasis.T                 # (N1, D), lag 0 first
    phi = phi / (phi.norm(dim=1, keepdim=True) + 1e-8)  # same as forward()
    return phi.flip(-1).detach().cpu().numpy()          # lag 0 last

def acceptance_kernel(sigma_taps, truncate=3.0):
    """Ommatidial acceptance function sampled on the tap lattice."""
    r = max(1, int(round(truncate * sigma_taps)))
    ax = np.arange(-r, r + 1)
    g = np.exp(-ax ** 2 / (2 * sigma_taps ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def effective_rf(cnn, dataset, truncate=3.0):
    """W1 convolved with the optics: the object comparable to a measured RF.

    The blur acted upstream of the network, so the learned weights carry no
    optics. Reverse-correlating the model against the *pre-blur* stimulus
    would recover this; convolving is exact because that path is linear.
    """
    sw = cnn.spatial_kernel[:, 0].detach().cpu().clone().numpy()
    sigma_taps = dataset.sigma_px / dataset.spacing_px
    k = acceptance_kernel(sigma_taps, truncate)
    eff = np.stack([convolve2d(w, k, mode="full") for w in sw])
    return eff, sigma_taps

def plot_cnn_subunits_1(cnn, dataset, effective=True, 
                        truncate=3.0, fs=12):
    n_su, max_delay = cnn.num_subunits_1, cnn.max_delay
    print(f"Plotting {n_su} subunits")
    tw = temporal_kernels(cnn)

    if effective:
        sw, sigma_taps = effective_rf(cnn, dataset, truncate)
        label = f"W₁ ⊛ O  (σ = {sigma_taps:.2f} taps)"
    else:
        sw = cnn.spatial_kernel[:, 0].detach().cpu().clone().numpy()
        label = "W₁ (weights only)"

    for n in range(n_su):                       # fix sign, push gain to time
        if abs(sw[n].min()) > sw[n].max():
            sw[n], tw[n] = -sw[n], -tw[n]
        s = np.linalg.norm(sw[n])
        sw[n] /= s
        tw[n] *= s

    k = sw.shape[-1]
    half = k * dataset.deg_per_tap / 2
    extent = [-half, half, -half, half]
    t_ms = np.arange(-max_delay + 1, 1) * (1000.0 / dataset.fps)
    vlim, ylim = np.abs(sw).max(), np.abs(tw).max()

    # fig, axs = plt.subplots(n_su, 3, figsize=(0.75*8, 0.75*3.2 * n_su), squeeze=False,
    #                         gridspec_kw=dict(width_ratios=[1, 1.9, 0.1]))
    # for n in range(n_su):
    #     axs[n, 0].plot(t_ms, tw[n])
    #     axs[n, 0].axhline(0, ls=":", c="k", lw=0.8)
    #     axs[n, 0].set_ylim(-ylim, ylim)
    #     axs[n, 0].tick_params(labelbottom=(n == n_su - 1), labelsize=fs)
    #     axs[n, 1].tick_params(labelsize=fs)
    #     im = axs[n, 1].imshow(sw[n], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
    #                           extent=extent, interpolation="nearest")
    #     axs[n, 1].set_title(f"subunit 1,{n+1}", fontsize = fs+2)
    #     plt.colorbar(im, cax=axs[n, 2])
    #     axs[n, 0].set_xlabel(r"$\Delta t$ [ms]", fontsize=fs)
    #     axs[n, 1].set_xlabel("Azimuth [deg]", fontsize = fs)
    #     axs[n, 1].set_ylabel("Elevation [deg]", fontsize = fs)
    
    # fig.suptitle(label, y=1.005)
    # fig.tight_layout()
    
    # return fig, axs
    fig = plt.figure(figsize=(0.75*8, 0.75*3.2*n_su), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(n_su, 1))
    axs = np.empty((n_su, 3), dtype=object)

    for n, sf in enumerate(subfigs):
        
        axs[n] = sf.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1.9, 0.1]))
        ax_t, ax_s, cax = axs[n]

        ax_t.plot(t_ms, tw[n])
        ax_t.axhline(0, ls=":", c="k", lw=0.8)
        ax_t.set_ylim(-ylim, ylim)
        ax_t.tick_params(labelbottom=(n == n_su - 1), labelsize=fs)
        ax_t.set_xlabel(r"$\Delta t$ [ms]", fontsize=fs)

        im = ax_s.imshow(sw[n], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                         extent=extent, interpolation="nearest")
        ax_s.tick_params(labelsize=fs)
        ax_s.set_xlabel("Azimuth [deg]", fontsize=fs)
        ax_s.set_ylabel("Elevation [deg]", fontsize=fs)
        sf.colorbar(im, cax=cax)
        l = axs[n, 0].get_position().x0
        r = axs[n, 1].get_position().x1  # use axs[n, 2] to include the colorbar
        sf.suptitle(f"Subunit 1,{n+1}", fontsize=fs+2.0, x=(l + r) / 2)
        
    fig.suptitle(label)
    return fig, axs



def plot_cnn_subunits_2(cnn: nn.Module):
    """
    Plot the second layer of subunits of a CNN.
    """
    cnn_filters_2 = cnn.layer2.weight.to("cpu").detach()

    fig, axs = plt.subplots(cnn.num_subunits_2,
                            cnn.num_subunits_1,
                            figsize=(4 * cnn.num_subunits_2,
                                    4 * cnn.num_subunits_1),
                            sharex=True, sharey=True)
    # vlim = abs(cnn_filters_2).max()
    percentile = 99
    vlim = torch.quantile(cnn_filters_2.abs().flatten(), percentile / 100).item()
    for i in range(cnn.num_subunits_2):
        for j in range(cnn.num_subunits_1):
            axs[i, j].imshow(cnn_filters_2[i, j],
                            vmin=-vlim, vmax=vlim, cmap="RdBu_r")

            axs[i, j].set_title('subunit 1,{} $\\to$ 2,{}'.format(j+1,i+1))
