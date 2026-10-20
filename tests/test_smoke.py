import torch
from motionnet.models import MotionCNN

def test_forward():
    m = MotionCNN(fps=150, rf_size_1=7, rf_size_2=5, num_subunits_1=5, num_subunits_2=8,
                  out_gain=1.0, initial_threshold=0.0, initial_gain=1.0,
                  input_noise_std=0.07, layer_noise_std=0.1)
    out = m(torch.randn(2, 80, 1, 31, 31))
    assert out.shape == (2, 80, 2)
    assert torch.isfinite(m._z_energy)