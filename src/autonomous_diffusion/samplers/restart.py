"""Restart sampling (Xu et al 2023).

Wraps an ODE base solver (EDM-Heun) with K restart cycles inside a
mid-noise band [sigma_lo, sigma_hi]. The base run integrates the full
trajectory only as far as sigma_lo (not down to zero); then each restart
cycle injects fresh noise of variance (sigma_hi^2 - sigma_lo^2) to lift x
back to the sigma_hi noise level and re-integrates down to sigma_lo;
finally a tail run takes x from sigma_lo to 0.

The Xu et al paper reports most of the FID gain comes from restarts placed
in the mid-noise regime where Heun's local truncation error peaks. Default
band is roughly [0.06, 1.0] which matches their CIFAR-10 table.

NFE accounting:
    base       = 2 * num_steps_base - 1
    per cycle  = 2 * inner_steps - 1
    tail       = 2 * tail_steps - 1
    total NFE  = base + K * per_cycle + tail
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


def _heun_through(net: Any, x: torch.Tensor, sigmas: torch.Tensor) -> tuple[torch.Tensor, int]:
    """EDM-Heun through sigmas[0] -> sigmas[-1]. Sigmas may or may not end at 0."""
    nfe = 0
    n = sigmas.shape[0] - 1
    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = denoise(net, x, sigma)
        d = (x - denoised) / sigma
        x_next = x + (sigma_next - sigma) * d
        nfe += 1
        if sigma_next.item() > 0:
            denoised2 = denoise(net, x_next, sigma_next)
            d2 = (x_next - denoised2) / sigma_next
            x_next = x + (sigma_next - sigma) * 0.5 * (d + d2)
            nfe += 1
        x = x_next
    return x, nfe


@register_sampler("restart")
class Restart(Sampler):
    """K restart cycles in a band [sigma_lo, sigma_hi]. Defaults follow the
    Xu et al CIFAR-10 recommendation: one restart in the mid-noise regime.
    """
    def __init__(
        self,
        num_restart: int = 1,
        inner_steps: int = 6,
        tail_steps: int = 4,
        sigma_lo: float = 0.06,
        sigma_hi: float = 1.0,
    ):
        self.num_restart = num_restart
        self.inner_steps = inner_steps
        self.tail_steps = tail_steps
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi

    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        sigma_min, sigma_max = resolve_sigma_range(net)
        shape = resolve_shape(net, image_shape)

        sigma_lo = max(self.sigma_lo, sigma_min)
        sigma_hi = min(self.sigma_hi, sigma_max)
        assert sigma_lo < sigma_hi, f"sigma_lo {sigma_lo} must be < sigma_hi {sigma_hi}"

        # Base grid: sigma_max -> sigma_lo (no trailing zero).
        base_full = karras_sigmas(num_steps, sigma_lo, sigma_max, device=device).to(torch.float32)
        base_sigmas = base_full[:-1]   # drop the appended 0; we stop at sigma_lo

        # Inner restart grid: sigma_hi -> sigma_lo (no trailing zero).
        inner_full = karras_sigmas(self.inner_steps, sigma_lo, sigma_hi, device=device).to(torch.float32)
        inner_sigmas = inner_full[:-1]

        # Tail: sigma_lo -> 0 (Karras with appended zero is exactly that).
        tail_sigmas = karras_sigmas(self.tail_steps, sigma_min, sigma_lo, device=device).to(torch.float32)

        var_inj = sigma_hi ** 2 - sigma_lo ** 2
        std_inj = float(max(0.0, var_inj)) ** 0.5

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(base_sigmas[0]), seed=seed + done, device=device)
            cur_nfe = 0
            gen_restart = torch.Generator(device=device).manual_seed(int(seed) + int(done) + 31337)

            # Base run: integrate to sigma_lo
            x, used = _heun_through(net, x, base_sigmas)
            cur_nfe += used

            # Restart cycles
            for _k in range(self.num_restart):
                if std_inj > 0:
                    noise = torch.randn(x.shape, generator=gen_restart, device=device, dtype=x.dtype)
                    x = x + std_inj * noise
                x, used = _heun_through(net, x, inner_sigmas)
                cur_nfe += used

            # Tail: sigma_lo -> 0
            x, used = _heun_through(net, x, tail_sigmas)
            cur_nfe += used

            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "schedule": "karras+restart",
                "solver": "edm_heun+restart",
                "num_steps_base": num_steps,
                "num_restart": self.num_restart,
                "inner_steps": self.inner_steps,
                "tail_steps": self.tail_steps,
                "sigma_band": [sigma_lo, sigma_hi],
            },
        )
