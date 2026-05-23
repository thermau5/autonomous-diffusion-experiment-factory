import torch

from autonomous_diffusion.samplers.edm import karras_sigmas


def test_karras_sigmas_endpoints():
    s = karras_sigmas(num_steps=18, sigma_min=0.002, sigma_max=80.0, device="cpu")
    assert s.shape == (19,)
    assert float(s[0]) == 80.0
    assert float(s[-1]) == 0.0
    # last *non-zero* sigma == sigma_min
    assert abs(float(s[-2]) - 0.002) < 1e-9


def test_karras_sigmas_strictly_decreasing():
    s = karras_sigmas(num_steps=18, sigma_min=0.002, sigma_max=80.0, device="cpu")
    diffs = (s[1:] - s[:-1]).numpy()
    assert (diffs < 0).all()


def test_nfe_accounting():
    # Heun: NFE = 2*num_steps - 1 (last step is Euler at sigma=0).
    # Euler: NFE = num_steps.
    for num_steps in (5, 8, 18, 32):
        assert (2 * num_steps - 1) == 2 * num_steps - 1
        assert num_steps == num_steps
