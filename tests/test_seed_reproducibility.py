"""Determinism with a toy net so we don't need EDM weights to run this test."""
import pytest
import torch

from autonomous_diffusion.critic.guards import check_seed_determinism
from autonomous_diffusion.samplers import get_sampler


class _ToyNet(torch.nn.Module):
    img_resolution = 8
    img_channels = 3
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_data = 0.5

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        torch.nn.init.zeros_(self.conv.weight)

    def forward(self, x, sigma, class_labels=None):
        # denoised estimate is just x scaled toward 0 by sigma_data^2/(sigma^2+sigma_data^2)
        s = sigma.view(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (s ** 2 + self.sigma_data ** 2)
        return c_skip * x + self.conv(x)


@pytest.mark.parametrize("sampler_id", ["edm_euler", "edm_heun"])
def test_same_seed_same_samples(sampler_id):
    net = _ToyNet().to("cpu").eval()
    for p in net.parameters():
        p.requires_grad_(False)
    samp = get_sampler(sampler_id)
    kwargs = dict(net=net, num_samples=4, num_steps=5, seed=42, device="cpu", batch_size=4)
    out_a = samp.sample(**kwargs)
    out_b = samp.sample(**kwargs)
    check_seed_determinism(out_a.samples, out_b.samples, sampler_id=sampler_id, seed=42)


@pytest.mark.parametrize("sampler_id", ["edm_euler", "edm_heun"])
def test_different_seed_different_samples(sampler_id):
    net = _ToyNet().to("cpu").eval()
    for p in net.parameters():
        p.requires_grad_(False)
    samp = get_sampler(sampler_id)
    out_a = samp.sample(net=net, num_samples=4, num_steps=5, seed=0, device="cpu", batch_size=4)
    out_b = samp.sample(net=net, num_samples=4, num_steps=5, seed=1, device="cpu", batch_size=4)
    assert not torch.equal(out_a.samples, out_b.samples)


def test_nfe_accounting_in_output():
    net = _ToyNet().to("cpu").eval()
    for p in net.parameters():
        p.requires_grad_(False)
    n_steps = 6
    out_euler = get_sampler("edm_euler").sample(
        net=net, num_samples=2, num_steps=n_steps, seed=0, device="cpu", batch_size=2,
    )
    out_heun = get_sampler("edm_heun").sample(
        net=net, num_samples=2, num_steps=n_steps, seed=0, device="cpu", batch_size=2,
    )
    assert out_euler.nfe == n_steps
    assert out_heun.nfe == 2 * n_steps - 1


@pytest.mark.parametrize("sampler_id", [
    "edm_euler", "edm_heun",
    "karras_schedule", "uniform_schedule",
    "ddim", "ddpm_ancestral",
    "dpm_solver", "dpm_solver_pp",
    "unipc", "deis", "pndm",
    "restart",
    "proposed_heun",
    "proposed_dpmpp",
    "proposed_unipc",
    "proposed_deis",
    "proposed_restart",
])
def test_every_sampler_smokes_on_toy_net(sampler_id):
    """Every registered baseline returns a tensor of the right shape on the toy
    net without raising. This catches obvious shape/dtype/registration bugs
    without needing the EDM pretrained weights."""
    from autonomous_diffusion.samplers import get_sampler
    net = _ToyNet().to("cpu").eval()
    for p in net.parameters():
        p.requires_grad_(False)
    out = get_sampler(sampler_id).sample(
        net=net, num_samples=2, num_steps=6, seed=0, device="cpu", batch_size=2,
    )
    assert out.samples.shape == (2, 3, 8, 8)
    assert out.samples.dtype == torch.float32
    # samples are clamped to [-1, 1] by the run driver
    assert out.samples.min().item() >= -1.0 - 1e-5
    assert out.samples.max().item() <= 1.0 + 1e-5
    assert out.nfe > 0
