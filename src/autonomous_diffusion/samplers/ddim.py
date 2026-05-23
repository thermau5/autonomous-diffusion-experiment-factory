"""DDIM (Song et al 2020).

Deterministic Euler step on the DDPM-beta-derived sigma schedule
(sigma_i = sqrt((1 - alpha_bar_i) / alpha_bar_i)). Same denoiser net as EDM;
the only difference vs. edm_euler is the sigma grid. NFE = num_steps.
"""
from __future__ import annotations

from typing import Any

import torch

from ._common import (
    ddpm_beta_sigmas,
    denoise,
    resolve_sigma_range,
    run_sampler,
)
from .base import Sampler, SamplerOutput, register_sampler


def _ddim_update(net: Any, x: torch.Tensor, i: int, sigmas: torch.Tensor, state: dict):
    sigma, sigma_next = sigmas[i], sigmas[i + 1]
    denoised = denoise(net, x, sigma)
    d = (x - denoised) / sigma
    return x + (sigma_next - sigma) * d, 1


@register_sampler("ddim")
class DDIM(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        sigma_min, sigma_max = resolve_sigma_range(net)
        sigmas = ddpm_beta_sigmas(
            num_steps,
            sigma_min_clamp=sigma_min,
            sigma_max_clamp=sigma_max,
            device=device,
        ).to(torch.float32)
        samples, nfe = run_sampler(
            net=net, sigmas=sigmas, update_fn=_ddim_update,
            num_samples=num_samples, seed=seed, device=device,
            batch_size=batch_size, image_shape=image_shape,
        )
        return SamplerOutput(samples=samples, nfe=nfe,
                             metadata={"schedule": "ddpm_linear", "solver": "ddim",
                                       "num_steps": num_steps})
