"""EDM samplers -- Karras 2022 Algorithm 1.

  - edm_euler  1st-order Euler steps,  NFE = num_steps
  - edm_heun   Heun 2nd-order corrector, NFE = 2*num_steps - 1
               (last step is Euler because sigma_N = 0)
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


def _edm_step(net: Any, x: torch.Tensor, sigma, sigma_next, *, heun: bool) -> tuple[torch.Tensor, int]:
    denoised = denoise(net, x, sigma)
    d = (x - denoised) / sigma
    x_next = x + (sigma_next - sigma) * d
    nfe = 1
    if heun and sigma_next.item() > 0:
        denoised2 = denoise(net, x_next, sigma_next)
        d2 = (x_next - denoised2) / sigma_next
        x_next = x + (sigma_next - sigma) * 0.5 * (d + d2)
        nfe += 1
    return x_next, nfe


def _edm_sample(
    *,
    net: Any,
    num_samples: int,
    num_steps: int,
    seed: int,
    device,
    batch_size: int,
    image_shape,
    heun: bool,
) -> SamplerOutput:
    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    sigma_min, sigma_max = resolve_sigma_range(net)
    sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)

    out: list[torch.Tensor] = []
    nfe_per_sample = 0
    done = 0
    while done < num_samples:
        b = min(batch_size, num_samples - done)
        x = sample_initial_noise((b, *shape), float(sigmas[0]), seed=seed + done, device=device)
        cur_nfe = 0
        for i in range(num_steps):
            x, used = _edm_step(net, x, sigmas[i], sigmas[i + 1], heun=heun)
            cur_nfe += used
        out.append(x.clamp(-1, 1).cpu())
        if done == 0:
            nfe_per_sample = cur_nfe
        done += b

    return SamplerOutput(
        samples=torch.cat(out, dim=0)[:num_samples],
        nfe=nfe_per_sample,
        metadata={"schedule": "karras", "sigma_min": sigma_min, "sigma_max": sigma_max,
                  "rho": 7.0, "num_steps": num_steps, "heun": heun},
    )


@register_sampler("edm_euler")
class EDMEuler(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        return _edm_sample(
            net=net, num_samples=num_samples, num_steps=num_steps, seed=seed,
            device=device, batch_size=batch_size, image_shape=image_shape, heun=False,
        )


@register_sampler("edm_heun")
class EDMHeun(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        return _edm_sample(
            net=net, num_samples=num_samples, num_steps=num_steps, seed=seed,
            device=device, batch_size=batch_size, image_shape=image_shape, heun=True,
        )
