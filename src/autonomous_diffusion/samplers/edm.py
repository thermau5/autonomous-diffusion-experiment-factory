"""EDM samplers — Karras 2022 Algorithm 1.

Two registered ids:
  - edm_euler   first-order Euler steps,        NFE = num_steps
  - edm_heun    Heun 2nd-order corrector,       NFE = 2*num_steps - 1
                (last step is Euler because sigma_{N} = 0).

The Karras noise schedule:
  sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
  sigma_N = 0
"""
from __future__ import annotations

from typing import Any

import torch

from .base import Sampler, SamplerOutput, register_sampler


def karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Length num_steps+1 tensor: sigma_0..sigma_{N-1}, then sigma_N=0."""
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([t, t.new_zeros([1])])  # last step ends at sigma=0


def _resolve_shape(net: Any, image_shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if image_shape is not None:
        return image_shape
    # EDM EDMPrecond exposes img_resolution and img_channels.
    res = getattr(net, "img_resolution", None)
    ch = getattr(net, "img_channels", None)
    if res is None or ch is None:
        raise ValueError(
            "image_shape not provided and net has no img_resolution/img_channels attrs"
        )
    return (ch, res, res)


def _edm_step(
    net: Any,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    heun: bool,
) -> tuple[torch.Tensor, int]:
    """One sampler step. Returns (x_next, nfe_consumed)."""
    sigma_b = sigma.expand(x.shape[0]).to(x.dtype)
    denoised = net(x, sigma_b).to(x.dtype)
    d = (x - denoised) / sigma
    x_next = x + (sigma_next - sigma) * d
    nfe = 1
    if heun and sigma_next.item() > 0:
        sigma_next_b = sigma_next.expand(x.shape[0]).to(x.dtype)
        denoised2 = net(x_next, sigma_next_b).to(x.dtype)
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
    device: str | torch.device,
    batch_size: int,
    image_shape: tuple[int, int, int] | None,
    heun: bool,
) -> SamplerOutput:
    device = torch.device(device)
    shape = _resolve_shape(net, image_shape)
    sigma_min = float(getattr(net, "sigma_min", 0.002))
    sigma_max = float(getattr(net, "sigma_max", 80.0))
    sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, device=device).to(torch.float32)

    g = torch.Generator(device=device).manual_seed(int(seed))
    out: list[torch.Tensor] = []
    nfe_per_sample = 0

    done = 0
    while done < num_samples:
        b = min(batch_size, num_samples - done)
        x = torch.randn((b, *shape), generator=g, device=device) * sigmas[0]
        cur_nfe = 0
        for i in range(num_steps):
            x, used = _edm_step(net, x, sigmas[i], sigmas[i + 1], heun=heun)
            cur_nfe += used
        out.append(x.clamp(-1, 1).cpu())
        if done == 0:
            nfe_per_sample = cur_nfe
        done += b

    samples = torch.cat(out, dim=0)[:num_samples]
    return SamplerOutput(
        samples=samples,
        nfe=nfe_per_sample,
        metadata={
            "schedule": "karras",
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "rho": 7.0,
            "num_steps": num_steps,
            "heun": heun,
        },
    )


@register_sampler("edm_euler")
class EDMEuler(Sampler):
    def sample(
        self,
        *,
        net,
        num_samples,
        num_steps,
        seed,
        device="cuda",
        batch_size=64,
        image_shape=None,
    ) -> SamplerOutput:
        return _edm_sample(
            net=net, num_samples=num_samples, num_steps=num_steps, seed=seed,
            device=device, batch_size=batch_size, image_shape=image_shape, heun=False,
        )


@register_sampler("edm_heun")
class EDMHeun(Sampler):
    def sample(
        self,
        *,
        net,
        num_samples,
        num_steps,
        seed,
        device="cuda",
        batch_size=64,
        image_shape=None,
    ) -> SamplerOutput:
        return _edm_sample(
            net=net, num_samples=num_samples, num_steps=num_steps, seed=seed,
            device=device, batch_size=batch_size, image_shape=image_shape, heun=True,
        )
