"""DPM-Solver (Lu et al 2022, first paper) -- 2nd-order singlestep.

Working entirely in sigma-space for the EDM/VE parameterization (alpha=1).
The probability-flow ODE is dx/dsigma = eps_theta(x, sigma), so the midpoint
(modified-Euler) method gives a 2nd-order step:

    sigma_mid = exp((log sigma_i + log sigma_next)/2)        # geometric mean
    eps_i   = (x_i - denoised(x_i, sigma_i)) / sigma_i
    u_i     = x_i + (sigma_mid - sigma_i) * eps_i            # half-step Euler
    eps_mid = (u_i - denoised(u_i, sigma_mid)) / sigma_mid
    x_next  = x_i + (sigma_next - sigma_i) * eps_mid         # full-step with midpoint slope

This is equivalent to DPM-Solver-2 singlestep (Lu et al Algorithm 1) for VE.
The `alpha_s/alpha_t` rescaling in their VP formulation is identically 1 for
VE, so it drops out. Final step to sigma_next=0 falls back to denoised.

NFE = 2*num_steps - 1.
"""
from __future__ import annotations

from typing import Any

import torch

from ._common import (
    denoise,
    karras_sigmas,
    resolve_sigma_range,
    run_sampler,
)
from .base import Sampler, SamplerOutput, register_sampler


def _dpm_solver_update(net: Any, x: torch.Tensor, i: int, sigmas: torch.Tensor, state: dict):
    sigma_i = sigmas[i]
    sigma_next = sigmas[i + 1]
    denoised_i = denoise(net, x, sigma_i)
    nfe = 1

    if sigma_next.item() == 0:
        return denoised_i, nfe

    # Geometric midpoint in sigma == arithmetic midpoint in lambda = -log sigma.
    sigma_mid = torch.exp(0.5 * (sigma_i.log() + sigma_next.log()))
    eps_i = (x - denoised_i) / sigma_i
    u = x + (sigma_mid - sigma_i) * eps_i
    denoised_mid = denoise(net, u, sigma_mid)
    nfe += 1
    eps_mid = (u - denoised_mid) / sigma_mid
    x_next = x + (sigma_next - sigma_i) * eps_mid
    return x_next, nfe


@register_sampler("dpm_solver")
class DPMSolver(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        sigma_min, sigma_max = resolve_sigma_range(net)
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)
        samples, nfe = run_sampler(
            net=net, sigmas=sigmas, update_fn=_dpm_solver_update,
            num_samples=num_samples, seed=seed, device=device,
            batch_size=batch_size, image_shape=image_shape,
        )
        return SamplerOutput(samples=samples, nfe=nfe,
                             metadata={"schedule": "karras", "solver": "dpm_solver_2_singlestep",
                                       "num_steps": num_steps, "order": 2})
