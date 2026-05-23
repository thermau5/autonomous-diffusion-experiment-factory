"""DEIS -- Diffusion Exponential Integrator Sampler (Zhang & Chen 2022).

Adams-Bashforth-style 2nd-order multistep on the eps-prediction. Working in
lambda = -log(sigma), with eps_i = (x_i - denoised_i)/sigma_i:

    h_i = lambda_{i+1} - lambda_i
    h_i_prev = lambda_i - lambda_{i-1}
    coef_cur  =  1 + h_i / (2 * h_i_prev)
    coef_prev =      -h_i / (2 * h_i_prev)
    eps_extrap = coef_cur * eps_i + coef_prev * eps_{i-1}
    x_{i+1} = (sigma_{i+1}/sigma_i) * x_i + (sigma_{i+1} - sigma_i) * eps_extrap

First step is 1st-order (no previous eps). Final step to sigma=0 also 1st-order.
NFE = num_steps.

This is the 'tAB-2' variant (time-adaptive Adams-Bashforth, 2nd order) which
matches the most common "DEIS" deployment in stable-diffusion-style stacks.
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


def _deis_update(net: Any, x: torch.Tensor, i: int, sigmas: torch.Tensor, state: dict):
    sigma_i = sigmas[i]
    sigma_next = sigmas[i + 1]
    denoised = denoise(net, x, sigma_i)
    eps_i = (x - denoised) / sigma_i

    if sigma_next.item() == 0:
        state["prev_eps"] = eps_i
        return denoised, 1

    if state.get("prev_eps") is None:
        # 1st-order Euler step in eps (sigma-space)
        x_next = x + (sigma_next - sigma_i) * eps_i
    else:
        # Adams-Bashforth-2 in lambda = -log sigma. The extrapolated eps is
        # used as the slope for an eps-space Euler step in sigma. For VE
        # parameterization there is no (sigma_next/sigma_i) rescaling on x.
        sigma_prev = sigmas[i - 1]
        lam_i = -sigma_i.log()
        lam_next = -sigma_next.log()
        lam_prev = -sigma_prev.log()
        h_i = lam_next - lam_i
        h_prev = lam_i - lam_prev
        coef_cur = 1 + h_i / (2 * h_prev)
        coef_prev = -h_i / (2 * h_prev)
        eps_extrap = coef_cur * eps_i + coef_prev * state["prev_eps"]
        x_next = x + (sigma_next - sigma_i) * eps_extrap

    state["prev_eps"] = eps_i
    return x_next, 1


@register_sampler("deis")
class DEIS(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        sigma_min, sigma_max = resolve_sigma_range(net)
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)
        samples, nfe = run_sampler(
            net=net, sigmas=sigmas, update_fn=_deis_update,
            num_samples=num_samples, seed=seed, device=device,
            batch_size=batch_size, image_shape=image_shape,
            state_factory=lambda: {"prev_eps": None},
        )
        return SamplerOutput(samples=samples, nfe=nfe,
                             metadata={"schedule": "karras", "solver": "deis_tAB2",
                                       "num_steps": num_steps, "order": 2})
