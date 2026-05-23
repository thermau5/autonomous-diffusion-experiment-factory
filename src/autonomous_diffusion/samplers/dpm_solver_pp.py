"""DPM-Solver++ 2M multistep (Lu et al 2022, "DPM-Solver++").

Working in lambda = -log(sigma) ("the half log-SNR"), the 2nd-order multistep
update on the x0-prediction (denoised) is

    h_i = lambda_{i+1} - lambda_i,
    r   = h_{i-1} / h_i,
    D_i = (1 + 1/(2r)) * denoised_i - (1/(2r)) * denoised_{i-1},
    x_{i+1} = (sigma_{i+1}/sigma_i) * x_i - (exp(-h_i) - 1) * D_i.

The first step is 1st-order (no previous denoised available); the final step
back to sigma=0 uses 1st-order to avoid a singular lambda. NFE = num_steps.

Uses the Karras sigma grid (rho=7) by default, which Lu et al's authors
recommend for low-NFE comparison.
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


def _dpmpp_update(net: Any, x: torch.Tensor, i: int, sigmas: torch.Tensor, state: dict):
    sigma_i = sigmas[i]
    sigma_next = sigmas[i + 1]
    denoised = denoise(net, x, sigma_i)

    if sigma_next.item() == 0 or state.get("prev_denoised") is None:
        # 1st-order step (first iter or last iter going to sigma=0)
        # x_next = (sigma_next/sigma_i) * x  -  (exp(-h_i) - 1) * denoised
        # with h_i = lambda_next - lambda_i = -log(sigma_next/sigma_i) when sigma_next > 0.
        if sigma_next.item() == 0:
            x_next = denoised
        else:
            h_i = (sigma_i.log() - sigma_next.log())   # = lambda_next - lambda_i
            x_next = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * denoised
    else:
        sigma_prev = sigmas[i - 1]
        h_i_1 = sigma_prev.log() - sigma_i.log()       # h_{i-1}
        h_i = sigma_i.log() - sigma_next.log()         # h_i
        r = h_i_1 / h_i
        D_i = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * state["prev_denoised"]
        x_next = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * D_i

    state["prev_denoised"] = denoised
    return x_next, 1


@register_sampler("dpm_solver_pp")
class DPMSolverPP(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        sigma_min, sigma_max = resolve_sigma_range(net)
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)
        samples, nfe = run_sampler(
            net=net, sigmas=sigmas, update_fn=_dpmpp_update,
            num_samples=num_samples, seed=seed, device=device,
            batch_size=batch_size, image_shape=image_shape,
            state_factory=lambda: {"prev_denoised": None},
        )
        return SamplerOutput(samples=samples, nfe=nfe,
                             metadata={"schedule": "karras", "solver": "dpm_solver_pp_2M",
                                       "num_steps": num_steps, "order": 2})
