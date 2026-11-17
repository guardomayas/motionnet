from motionnet.models import makeRaisedCosBasis
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def unit(w, dims):
    """w scaled to unit L2 norm over dims. forward() sees only this, so raw norms are free."""
    return w / (torch.linalg.vector_norm(w, dim=dims, keepdim=True) + 1e-8)


class MotionCNN(nn.Module):
    """Two-layer motion CNN.

    Layer 1: separable space x time filters, gain g_c, threshold theta_c, ReLU,
             additive output noise (per frame).
    Layer 2: spatial conv across layer-1 channels, frame by frame, gain h_m,
             threshold theta2_m, ReLU, multiplicative output noise (per frame).
    Readout: fixed-scale linear map per location, then spatial mean.

    Activity costs (Karklin & Simoncelli convention): L1 rate after the
    nonlinearity, summed over channels, averaged over batch/time/space,
    taken before noise. Exposed as self._rate1, self._rate2.
    """

    def __init__(self,
                 fps=None,
                 rf_size_1=None,
                 rf_size_2=None,
                 num_subunits_1=None,
                 num_subunits_2=None,
                 out_gain=None,
                 initial_threshold=None,
                 initial_gain=None,
                 input_noise_std=None,
                 layer_noise_std=None,
                 layer2_noise_std=0.0,
                 noise_f_ref=75.0,
                 nonlinearity="relu",
                 readout_bias=True,
                 zero_dc2=False
                 ):
        super().__init__()

        self.rf_size_1 = rf_size_1
        self.rf_size_2 = rf_size_2
        self.num_subunits_1 = num_subunits_1
        self.num_subunits_2 = num_subunits_2
        self.nonlinearity = nonlinearity

        # -------------------------------------------------------
        # Noise
        # -------------------------------------------------------
        # Input noise is defined at noise_f_ref and scaled by sqrt(fps / f_ref):
        # phi_c integrates over a fixed time, so this keeps the filtered SNR
        # independent of frame rate.
        self.input_noise_std = input_noise_std * math.sqrt(fps / noise_f_ref)
        # Layer noises are per frame: nothing downstream integrates over time.
        self.layer_noise_std = layer_noise_std      # additive, layer 1
        self.layer2_noise_std = layer2_noise_std    # multiplicative, layer 2

        # =======================================================
        # Layer 1: separable spatio-temporal subunits
        # =======================================================
        self.spatial_kernel = nn.Parameter(
            torch.randn(num_subunits_1, 1, rf_size_1, rf_size_1)
        )

        dt = 1 / fps
        nbasis = 8
        RFstart = 15e-3
        RFend = 130e-3
        b = 0.02

        t, B_orth, B = makeRaisedCosBasis(nbasis, dt, [RFstart, RFend], b, zflag=0)
        D = B.shape[0]
        print(f"support: {D} frames = {t[-1]*1e3:.0f} ms")
        print("rank:", B_orth.shape[1], "of", nbasis)
        print("condition number:", np.linalg.cond(B))

        self.register_buffer("tbasis", torch.as_tensor(B, dtype=torch.float32))  # (D, nb), lag 0 first
        self.max_delay = self.tbasis.shape[0]
        self.temporal_w = nn.Parameter(0.1 * torch.randn(num_subunits_1, B.shape[1]))

        self.threshold = nn.Parameter(
            torch.full((num_subunits_1,), float(initial_threshold))
        )
        raw_initial_gain = torch.log(torch.expm1(torch.tensor(float(initial_gain))))
        self.raw_gain = nn.Parameter(raw_initial_gain.repeat(num_subunits_1))

        # =======================================================
        # Layer 2: rectified spatial pooling across channels
        # =======================================================
        self.layer2 = nn.Conv2d(
            in_channels=num_subunits_1,
            out_channels=num_subunits_2,
            kernel_size=rf_size_2,
            padding=0,
            bias=False,          # the threshold plays this role
        )
        self.threshold2 = nn.Parameter(
            torch.full((num_subunits_2,), float(initial_threshold))
        )
        self.raw_gain2 = nn.Parameter(
            torch.full((num_subunits_2,), math.log(math.e - 1))   # softplus -> 1
        )

        # =======================================================
        # Readout
        # =======================================================
        self.readout = nn.Parameter(0.1 * torch.randn(2, num_subunits_2))   # rows: v_x, v_y
        self.zero_dc2 = zero_dc2  # if True, layer-2 kernels are forced to have zero DC (mean) over space
        if readout_bias:
            self.readout_bias = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("readout_bias", torch.zeros(2))            # fixed at 0, not trained
        self.out_gain = out_gain
        self.out_gain = out_gain                                            # fixed float, NOT a Parameter

        with torch.no_grad():
            self.layer2.weight -= self.layer2.weight.mean(dim=(2, 3), keepdim=True)
            # Start raw norms at 1: Adam's step size relative to the weights depends on them
            for w, dims in ((self.spatial_kernel, (1, 2, 3)), (self.temporal_w, (1,)),
                            (self.layer2.weight, (1, 2, 3)), (self.readout, (1,))):
                w.copy_(unit(w, dims))

    # -----------------------------------------------------------
    # Parameters in constrained form
    # -----------------------------------------------------------
    @property
    def gain(self):
        """Layer-1 gains g_c > 0, shape (num_subunits_1,)."""
        return F.softplus(self.raw_gain)

    @property
    def gain2(self):
        """Layer-2 gains h_m > 0, shape (num_subunits_2,)."""
        return F.softplus(self.raw_gain2)

    def _nl(self, x):
        if self.nonlinearity == "relu":
            return F.relu(x)
        if self.nonlinearity == "softplus":
            return F.softplus(x)
        raise ValueError(f"unknown nonlinearity {self.nonlinearity!r}")

    def add_input_noise(self, movie):
        if self.input_noise_std > 0:
            movie = movie + self.input_noise_std * torch.randn_like(movie)
        return movie
    def layer2_kernel(self):
        """Effective layer-2 kernels as used in forward(): optionally zero-DC per input channel, then unit norm."""
        K = self.layer2.weight
        if self.zero_dc2:
            K = K - K.mean(dim=(2, 3), keepdim=True)
        return unit(K, (1, 2, 3))

    # -----------------------------------------------------------
    # Forward
    # -----------------------------------------------------------
    def forward(self, movie):
        movie = self.add_input_noise(movie)
        B, T, C, H, W = movie.shape

        # 1. Spatial conv per frame
        x = movie.reshape(B * T, C, H, W)
        x = F.conv2d(x, unit(self.spatial_kernel, (1, 2, 3)))
        _, N1, H1, W1 = x.shape
        x = x.reshape(B, T, N1, H1, W1)

        # 2. Causal temporal conv at each spatial location
        x = x.permute(0, 3, 4, 2, 1).reshape(B * H1 * W1, N1, T)
        x = F.pad(x, (self.max_delay - 1, 0))
        phi = unit(self.temporal_w @ self.tbasis.T, (1,))          # (N1, D)
        x = F.conv1d(x, phi.flip(-1)[:, None, :], groups=N1)
        x = x.reshape(B, H1, W1, N1, T).permute(0, 4, 3, 1, 2)     # (B, T, N1, H1, W1)

        # Layer-1 nonlinearity
        z = x                                                      # pre-threshold drive
        g = self.gain[None, None, :, None, None]
        th = self.threshold[None, None, :, None, None]
        x = g * self._nl(z - th)

        self._rate1 = x.mean(dim=(0, 1, 3, 4)).sum()               # L1 rate cost, before noise
        with torch.no_grad():                                      # diagnostics
            self._energy1_c = (g * z).pow(2).mean(dim=(0, 1, 3, 4))
            self._energy1 = self._energy1_c.sum()
            self._l1_std = x.std(dim=(0, 1, 3, 4))
            self._active1 = (x > 0).float().mean(dim=(0, 1, 3, 4))

        if self.layer_noise_std > 0:
            x = x + self.layer_noise_std * torch.randn_like(x)     # additive

        # 3. Layer 2: spatial conv across channels, frame by frame
        x = x.reshape(B * T, N1, H1, W1)
        K = self.layer2_kernel()
        h = self.gain2[None, :, None, None]
        drive = F.conv2d(x, K)                                     # pre-threshold drive
        s = drive - self.threshold2[None, :, None, None]
        x = h * self._nl(s)

        self._rate2 = x.mean(dim=(0, 2, 3)).sum()                  # L1 rate cost, before noise
        with torch.no_grad():
            self._active2 = (x > 0).float().mean(dim=(0, 2, 3))
            self._rate2_c = x.mean(dim=(0, 2, 3))                  # per-unit <y>
            self._y2sq_c = x.pow(2).mean(dim=(0, 2, 3))            # per-unit <y^2>    
            self._drive_sq = drive.pow(2).mean(dim=(-1, -2))       # (B*T, C2): linear drive energy per frame
            self._drive_mean = drive.mean(dim=(-1, -2))            # (B*T, C2)
            self._drive_rect = F.relu(drive).mean(dim=(-1, -2))    # (B*T, C2): drive rectified at 0

        if self.layer2_noise_std > 0:
            # multiplicative: adds ~ sum_m (R_xm^2 + R_ym^2) <y_m^2> to the loss,
            # i.e. an L2 cost on activity weighted by the readout
            x = x * (1 + self.layer2_noise_std * torch.randn_like(x))

        _, N2, H2, W2 = x.shape

        # 4. Linear readout per location, then pool over space
        self._pool = x.mean(dim=(-1, -2)).detach()                 # (B*T, N2), diagnostics
        R = unit(self.readout, (1,))                               # unit rows
        local = self.out_gain * torch.einsum("bnhw,kn->bhwk", x, R) + self.readout_bias
        self._local_v = local.reshape(B, T, H2 * W2, 2)            # (B, T, N, 2)
        return self._local_v.mean(2)