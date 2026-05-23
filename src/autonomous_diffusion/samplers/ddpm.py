"""DDPM ancestral sampling on a DDPM-beta-derived sigma schedule.

Same denoiser net; we run the standard ancestral update in sigma-space:
    sigma_up^2  = (sigma_next^2 / sigma_cur^2) * (sigma_cur^2 - sigma_next^2)
    sigma_down  = sqrt(sigma_next^2 - sigma_up^2)
    x_next      = denoised + sigma_down/sigma_cur * (x - denoised) + sigma_up * noise

NFE = num_steps.
"""
from __future__ import annotations

import torch

from ._common import (
    ancestral_step_sigmas,
    ddpm_beta_sigmas,
    denoise,
    resolve_shape,
    resolve_sigma_range,
    sample_initial_noise,
)
from .base import Sampler, SamplerOutput, register_sampler


@register_sampler("ddpm_ancestral")
class DDPMAncestral(Sampler):
    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        sigma_min, sigma_max = resolve_sigma_range(net)
        sigmas = ddpm_beta_sigmas(
            num_steps, sigma_min_clamp=sigma_min, sigma_max_clamp=sigma_max,
            device=device,
        ).to(torch.float32)
        shape = resolve_shape(net, image_shape)

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(sigmas[0]), seed=seed + done, device=device)
            # Separate generator for ancestral noise injection so it doesn't share
            # state with the initial-noise generator.
            gen = torch.Generator(device=device).manual_seed(int(seed) + int(done) + 7919)
            cur_nfe = 0
            for i in range(num_steps):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]
                denoised = denoise(net, x, sigma)
                cur_nfe += 1
                if sigma_next.item() == 0:
                    x = denoised
                    continue
                sigma_down, sigma_up = ancestral_step_sigmas(float(sigma), float(sigma_next), eta=1.0)
                d = (x - denoised) / sigma
                x = x + (sigma_down - float(sigma)) * d
                if sigma_up > 0:
                    noise = torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype)
                    x = x + sigma_up * noise
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={"schedule": "ddpm_linear", "solver": "ddpm_ancestral",
                      "num_steps": num_steps, "eta": 1.0},
        )
