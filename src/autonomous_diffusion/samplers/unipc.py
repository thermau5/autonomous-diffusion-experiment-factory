"""UniPC (Zhao et al 2023) -- 2nd-order predictor-corrector.

Predictor is DPM-Solver++ 2M style: use prior denoised D_{i-1} to extrapolate
a 2nd-order step. After taking the predicted x_{i+1}, evaluate the denoiser
again at sigma_{i+1} (this becomes denoised_{i+1}) and use it to *correct*
the step at no extra NFE cost (since denoised_{i+1} is the same call we'd
make at the start of the next predictor step). Net NFE = num_steps.

Working in lambda = -log(sigma):
  h_i        = lambda_{i+1} - lambda_i
  r          = (lambda_i - lambda_{i-1}) / h_i
  D_i^p      = (1 + 1/(2r)) * denoised_i - (1/(2r)) * denoised_{i-1}        (predictor)
  x_{i+1}^p  = (sigma_{i+1}/sigma_i) * x_i - (exp(-h_i) - 1) * D_i^p
  denoised_{i+1} = net(x_{i+1}^p, sigma_{i+1})                              (reused next step)
  D_i^c      = (1/2) * denoised_i + (1/2) * denoised_{i+1}                  (corrector)
  x_{i+1}    = (sigma_{i+1}/sigma_i) * x_i - (exp(-h_i) - 1) * D_i^c

First step is 1st-order. Final step to sigma=0 also 1st-order.
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


@register_sampler("unipc")
class UniPC(Sampler):
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
            # Pre-compute denoised at sigma_0 once
            denoised_cur = denoise(net, x, sigmas[0])
            cur_nfe += 1
            denoised_prev = None

            for i in range(num_steps):
                sigma_i = sigmas[i]
                sigma_next = sigmas[i + 1]

                if sigma_next.item() == 0:
                    # Boundary: 1st-order
                    x = denoised_cur
                    break

                # PREDICTOR
                if denoised_prev is None:
                    # 1st-order start
                    h_i = sigma_i.log() - sigma_next.log()
                    x_pred = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * denoised_cur
                else:
                    sigma_prev = sigmas[i - 1]
                    h_i_1 = sigma_prev.log() - sigma_i.log()
                    h_i = sigma_i.log() - sigma_next.log()
                    r = h_i_1 / h_i
                    D_pred = (1 + 1 / (2 * r)) * denoised_cur - (1 / (2 * r)) * denoised_prev
                    x_pred = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * D_pred

                # CORRECTOR: get denoised at the predicted point and combine
                denoised_next = denoise(net, x_pred, sigma_next)
                cur_nfe += 1
                D_corr = 0.5 * denoised_cur + 0.5 * denoised_next
                h_i_eff = sigma_i.log() - sigma_next.log()
                x = (sigma_next / sigma_i) * x - torch.expm1(-h_i_eff) * D_corr

                denoised_prev = denoised_cur
                denoised_cur = denoised_next  # already evaluated, reuse next iter

            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={"schedule": "karras", "solver": "unipc_2pc",
                      "num_steps": num_steps, "order": 2},
        )
