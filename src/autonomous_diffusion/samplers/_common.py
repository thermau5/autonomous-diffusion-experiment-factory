"""Shared sigma-grid + denoiser helpers used by every sampler.

All samplers consume a pretrained EDM `EDMPrecond` (or compatible) net and
denoise via `net(x, sigma) -> x0_estimate`. Differences between samplers live
in (a) the sigma grid and (b) the step update rule -- both are clearly
factored here so the math in each sampler module reads cleanly against the
original paper.
"""
from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Net introspection
# ---------------------------------------------------------------------------

def resolve_shape(net: Any, image_shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if image_shape is not None:
        return image_shape
    res = getattr(net, "img_resolution", None)
    ch = getattr(net, "img_channels", None)
    if res is None or ch is None:
        raise ValueError(
            "image_shape not provided and net has no img_resolution/img_channels attrs"
        )
    return (ch, res, res)


def resolve_sigma_range(net: Any) -> tuple[float, float]:
    """Clamp net.sigma_min/max to Karras's EDM defaults (0.002, 80).

    EDMPrecond exposes 0/inf as "use sampler defaults" placeholders; VP/VE
    expose finite values from training. NVlabs/edm/generate.py does the same
    max/min clamp.
    """
    default_min, default_max = 0.002, 80.0
    net_min = float(getattr(net, "sigma_min", default_min) or default_min)
    net_max = float(getattr(net, "sigma_max", default_max))
    if not (0 < net_max < float("inf")):
        net_max = default_max
    return max(default_min, net_min), min(default_max, net_max)


# ---------------------------------------------------------------------------
# Sigma grids -- each returns a length (num_steps+1) tensor ending at sigma=0
# ---------------------------------------------------------------------------

def karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([t, t.new_zeros([1])])


def uniform_log_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Uniform spacing in log-sigma. ('uniform' baseline arm.)"""
    log_lo, log_hi = torch.log(torch.tensor(sigma_min, dtype=dtype, device=device)), \
                     torch.log(torch.tensor(sigma_max, dtype=dtype, device=device))
    t = torch.exp(torch.linspace(float(log_hi), float(log_lo), num_steps, dtype=dtype, device=device))
    return torch.cat([t, t.new_zeros([1])])


def ddpm_beta_sigmas(
    num_steps: int,
    *,
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    schedule: str = "linear",
    sigma_min_clamp: float = 0.002,
    sigma_max_clamp: float = 80.0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """DDPM beta-schedule sigmas (Ho et al / Song & Ermon convention),
    subsampled to `num_steps` and clamped to the EDM sigma range.

    sigma_i = sqrt((1 - alpha_bar_i) / alpha_bar_i)
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=dtype, device=device)
    elif schedule == "cosine":
        # Nichol & Dhariwal 2021 cosine schedule
        s = 0.008
        steps = torch.arange(num_train_timesteps + 1, dtype=dtype, device=device) / num_train_timesteps
        f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-8)
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, 0.999)
    else:
        raise ValueError(f"unknown ddpm schedule {schedule!r}")
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    full_sigmas = torch.sqrt((1.0 - alpha_bar) / alpha_bar).clamp(sigma_min_clamp, sigma_max_clamp)
    # Subsample monotonically from sigma_max -> sigma_min
    # Pick num_steps indices spanning the full schedule.
    if num_steps >= num_train_timesteps:
        sub = full_sigmas
    else:
        idx = torch.linspace(num_train_timesteps - 1, 0, num_steps, dtype=torch.float64, device=device).round().long()
        sub = full_sigmas[idx]
    sub = sub.flip(0) if sub[0] < sub[-1] else sub
    sub = torch.clamp(sub, sigma_min_clamp, sigma_max_clamp)
    return torch.cat([sub, sub.new_zeros([1])])


# ---------------------------------------------------------------------------
# Denoiser convenience -- every sampler should funnel through these helpers
# ---------------------------------------------------------------------------

def denoise(net: Any, x: torch.Tensor, sigma) -> torch.Tensor:
    """net(x, sigma)-style denoising: returns x0 estimate at noise level sigma."""
    if isinstance(sigma, (float, int)):
        sigma_b = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
    elif sigma.dim() == 0:
        sigma_b = sigma.expand(x.shape[0]).to(x.dtype)
    else:
        sigma_b = sigma.to(x.dtype)
    return net(x, sigma_b).to(x.dtype)


def eps_from_denoised(x: torch.Tensor, denoised: torch.Tensor, sigma) -> torch.Tensor:
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    sigma_view = sigma.view(-1, *([1] * (x.dim() - 1))) if sigma.dim() > 0 else sigma
    return (x - denoised) / sigma_view


def sample_initial_noise(
    shape: tuple[int, ...],
    sigma_init: float,
    *,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(shape, generator=g, device=device, dtype=dtype) * sigma_init


def run_sampler(
    *,
    net: Any,
    sigmas: torch.Tensor,
    update_fn,
    num_samples: int,
    seed: int,
    device,
    batch_size: int,
    image_shape: tuple[int, int, int] | None,
    state_factory=lambda: {},
) -> tuple[torch.Tensor, int]:
    """Generic batched driver.

    `update_fn(net, x, i, sigmas, state) -> (x_next, nfe_consumed)`. State is
    a per-batch dict for multistep samplers to stash prior denoised values.
    Returns (samples, nfe_per_sample). `nfe_per_sample` is the total NFE on
    the first batch (deterministic across batches by construction).
    """
    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    num_steps = sigmas.shape[0] - 1

    out: list[torch.Tensor] = []
    nfe_per_sample = 0
    done = 0
    while done < num_samples:
        b = min(batch_size, num_samples - done)
        x = sample_initial_noise((b, *shape), float(sigmas[0]), seed=seed + done, device=device)
        cur_nfe = 0
        state = state_factory()
        for i in range(num_steps):
            x, used = update_fn(net, x, i, sigmas, state)
            cur_nfe += used
        out.append(x.clamp(-1, 1).cpu())
        if done == 0:
            nfe_per_sample = cur_nfe
        done += b

    return torch.cat(out, dim=0)[:num_samples], nfe_per_sample


def ancestral_step_sigmas(sigma_from, sigma_to, eta: float = 1.0):
    """k-diffusion-style ancestral split: returns (sigma_down, sigma_up) s.t.
    sigma_down^2 + sigma_up^2 = sigma_to^2 and sigma_up tracks the DDPM
    variance schedule.
    """
    if eta == 0 or sigma_to == 0:
        return sigma_to, sigma_to * 0
    sigma_to_sq = sigma_to ** 2
    sigma_from_sq = sigma_from ** 2
    sigma_up_sq = eta * eta * sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq
    sigma_up_sq = max(0.0, float(sigma_up_sq))
    sigma_up = min(float(sigma_to), sigma_up_sq ** 0.5)
    sigma_down = max(0.0, float(sigma_to_sq - sigma_up ** 2)) ** 0.5
    return sigma_down, sigma_up
