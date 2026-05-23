"""Pure schedule baselines.

Both use the EDM-Heun second-order corrector to isolate the contribution of
the *schedule* (sigma grid) from the solver. Distinct sigma grids:
  - karras_schedule    rho=7 Karras spacing (Karras 2022)
  - uniform_schedule   uniform spacing in log-sigma
"""
from __future__ import annotations

from typing import Any

import torch

from ._common import (
    denoise,
    karras_sigmas,
    resolve_sigma_range,
    run_sampler,
    uniform_log_sigmas,
)
from .base import Sampler, SamplerOutput, register_sampler


def _heun_update(net: Any, x: torch.Tensor, i: int, sigmas: torch.Tensor, state: dict):
    sigma, sigma_next = sigmas[i], sigmas[i + 1]
    denoised = denoise(net, x, sigma)
    d = (x - denoised) / sigma
    x_next = x + (sigma_next - sigma) * d
    nfe = 1
    if sigma_next.item() > 0:
        denoised2 = denoise(net, x_next, sigma_next)
        d2 = (x_next - denoised2) / sigma_next
        x_next = x + (sigma_next - sigma) * 0.5 * (d + d2)
        nfe += 1
    return x_next, nfe


def _run(name: str, sigmas_fn, *, net, num_samples, num_steps, seed, device, batch_size, image_shape):
    sigma_min, sigma_max = resolve_sigma_range(net)
    sigmas = sigmas_fn(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)
    samples, nfe = run_sampler(
        net=net, sigmas=sigmas, update_fn=_heun_update,
        num_samples=num_samples, seed=seed, device=device,
        batch_size=batch_size, image_shape=image_shape,
    )
    return SamplerOutput(samples=samples, nfe=nfe,
                         metadata={"schedule": name, "sigma_min": sigma_min, "sigma_max": sigma_max,
                                   "num_steps": num_steps, "solver": "heun"})


@register_sampler("karras_schedule")
class KarrasSchedule(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        return _run("karras", karras_sigmas,
                    net=net, num_samples=num_samples, num_steps=num_steps, seed=seed,
                    device=device, batch_size=batch_size, image_shape=image_shape)


@register_sampler("uniform_schedule")
class UniformSchedule(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        return _run("uniform_log", uniform_log_sigmas,
                    net=net, num_samples=num_samples, num_steps=num_steps, seed=seed,
                    device=device, batch_size=batch_size, image_shape=image_shape)
