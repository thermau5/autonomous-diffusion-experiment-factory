"""PNDM / PLMS (Liu et al 2022) -- Pseudo Numerical Methods for Diffusion.

Architecture: warm up with 3 Runge-Kutta-4 (RK4) steps to fill an eps history
of length 4, then proceed with Adams-Bashforth-4 (PLMS) for the remaining
steps. The 4th-order linear multistep update on eps in sigma-space is:

    eps_AB4 = (55 eps_i - 59 eps_{i-1} + 37 eps_{i-2} - 9 eps_{i-3}) / 24
    x_{i+1} = (sigma_{i+1}/sigma_i) * x_i + (sigma_{i+1} - sigma_i) * eps_AB4

Each RK4 warmup step costs 4 NFE (so warmup = 12 NFE for 3 steps); each PLMS
step costs 1 NFE. Final step to sigma=0 falls back to Euler in eps.

NFE = 12 + max(0, num_steps - 3) for num_steps >= 4, otherwise 4 * num_steps.
"""
from __future__ import annotations

from typing import Any

import torch

from ._common import (
    denoise,
    karras_sigmas,
    resolve_shape,
    resolve_sigma_range,
    sample_initial_noise,
)
from .base import Sampler, SamplerOutput, register_sampler


def _eps(net: Any, x: torch.Tensor, sigma) -> torch.Tensor:
    denoised = denoise(net, x, sigma)
    return (x - denoised) / sigma


def _rk4_step(net: Any, x: torch.Tensor, sigma, sigma_next):
    """Classical RK4 on dx/dlambda = -sigma * eps where d sigma / d lambda = -sigma.
    Implemented as RK4 in sigma directly with d_i = (x - denoised)/sigma."""
    h = sigma_next - sigma
    sigma_mid = (sigma + sigma_next) * 0.5

    k1 = _eps(net, x, sigma)
    k2 = _eps(net, x + 0.5 * h * k1, sigma_mid)
    k3 = _eps(net, x + 0.5 * h * k2, sigma_mid)
    k4 = _eps(net, x + h * k3, sigma_next)
    eps_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x_next = x + h * eps_avg
    return x_next, k1, 4   # store k1 = eps at sigma_i for later AB4


@register_sampler("pndm")
class PNDM(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        sigma_min, sigma_max = resolve_sigma_range(net)
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)
        shape = resolve_shape(net, image_shape)

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(sigmas[0]), seed=seed + done, device=device)
            cur_nfe = 0
            eps_hist: list[torch.Tensor] = []

            for i in range(num_steps):
                sigma_i = sigmas[i]
                sigma_next = sigmas[i + 1]

                if sigma_next.item() == 0:
                    denoised = denoise(net, x, sigma_i)
                    cur_nfe += 1
                    x = denoised
                    break

                if i < 3:
                    # RK4 warmup
                    x, eps_at_i, used = _rk4_step(net, x, sigma_i, sigma_next)
                    cur_nfe += used
                    eps_hist.append(eps_at_i)
                else:
                    # PLMS (Adams-Bashforth-4) in sigma-space, VE parameterization.
                    # No (sigma_next/sigma_i) rescaling on x.
                    eps_at_i = _eps(net, x, sigma_i)
                    cur_nfe += 1
                    eps_hist.append(eps_at_i)
                    e = eps_hist
                    eps_ab4 = (55 * e[-1] - 59 * e[-2] + 37 * e[-3] - 9 * e[-4]) / 24
                    x = x + (sigma_next - sigma_i) * eps_ab4
                    if len(eps_hist) > 4:
                        eps_hist = eps_hist[-4:]

            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={"schedule": "karras", "solver": "pndm_plms_ab4",
                      "num_steps": num_steps, "order": 4},
        )
