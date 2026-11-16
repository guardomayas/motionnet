import math
import torch
from motionnet.models import MotionCNN


def make(**kw):
    args = dict(fps=150, rf_size_1=7, rf_size_2=5, num_subunits_1=5, num_subunits_2=8,
                out_gain=1.0, initial_threshold=0.0, initial_gain=1.0,
                input_noise_std=0.10, layer_noise_std=0.10, layer2_noise_std=0.1,
                noise_f_ref=75.0)
    args.update(kw)
    return MotionCNN(**args)


def test_forward_and_costs():
    m = make()
    out = m(torch.randn(2, 80, 1, 31, 31))
    assert out.shape == (2, 80, 2)
    assert m._local_v.shape == (2, 80, 21 * 21, 2)        # 31 -> 25 -> 21 with 7x7, 5x5
    for name in ("_rate1", "_rate2", "_energy1"):
        assert torch.isfinite(getattr(m, name)), name


def test_input_noise_scaling():
    assert math.isclose(make(fps=150).input_noise_std, 0.10 * math.sqrt(2), rel_tol=1e-6)
    assert math.isclose(make(fps=75).input_noise_std, 0.10, rel_tol=1e-6)


def test_every_parameter_gets_gradient():
    m = make()
    out = m(torch.randn(2, 60, 1, 31, 31))
    (out.pow(2).mean() + m._rate1 + m._rate2).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient: {missing}"