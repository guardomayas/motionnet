# src/motionnet/train/losses.py

import torch
import torch.nn.functional as F

### PENALTIES
_LAPLACIAN = torch.tensor([[0., 1., 0.],
                           [1., -4., 1.],
                           [0., 1., 0.]]).view(1, 1, 3, 3)


def laplacian_penalty(weight):
    """Mean squared spatial Laplacian of a conv kernel.

    weight: (out, in, k, k). Penalises high spatial frequency in the filter
    itself -- frequencies the optics already removed from the input, so they
    carry no signal and are otherwise unconstrained.
    """
    w = weight.flatten(0, 1).unsqueeze(1)                 # (out*in, 1, k, k)
    lap = _LAPLACIAN.to(weight.device, weight.dtype)
    return F.conv2d(w, lap).pow(2).mean()
