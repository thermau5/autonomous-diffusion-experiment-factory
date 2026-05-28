"""UniPC 2PC on the finite-N calibrated certificate grid (PDF v4.1, App. C).

Companion to the locked `proposed_unipc` (pointwise m_s*). Runs the UniPC
predictor-corrector solver on the calibrated sequence grid (shares the
2M-predictor one-step error surface with DPM-Solver++). Reporting both
quantifies the A approximately B gap for the headline UniPC core.

NFE per sample = num_steps.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import denoise, resolve_shape, sample_initial_noise
from ._seq_calib import get_or_build_seq_grid
from .base import Sampler, SamplerOutput, register_sampler


@register_sampler("proposed_unipc_seq")
class ProposedUniPCSeq(Sampler):
    CORE = "unipc"

    def __init__(self, *, cache_root: str | Path = "outputs/calibration"):
        self.cache_root = Path(cache_root)

    def sample(self, *, net, num_samples, num_steps, seed,
               device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        shape = resolve_shape(net, image_shape)
        grid = get_or_build_seq_grid(net, num_steps, self.CORE, root=self.cache_root,
                                     device=device, image_shape=image_shape)
        step_sigmas = torch.tensor(grid, dtype=torch.float32, device=device)

        out = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(step_sigmas[0]), seed=seed + done, device=device)
            cur_nfe = 0
            denoised_cur = denoise(net, x, step_sigmas[0]); cur_nfe += 1
            denoised_prev = None
            for i in range(num_steps):
                sigma_i = step_sigmas[i]; sigma_next = step_sigmas[i + 1]
                if sigma_next.item() == 0:
                    x = denoised_cur
                    break
                if denoised_prev is None:
                    h_i = sigma_i.log() - sigma_next.log()
                    x_pred = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * denoised_cur
                else:
                    sigma_prev = step_sigmas[i - 1]
                    h_i_1 = sigma_prev.log() - sigma_i.log()
                    h_i = sigma_i.log() - sigma_next.log()
                    r = h_i_1 / h_i
                    D_pred = (1 + 1 / (2 * r)) * denoised_cur - (1 / (2 * r)) * denoised_prev
                    x_pred = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * D_pred
                denoised_next = denoise(net, x_pred, sigma_next); cur_nfe += 1
                D_corr = 0.5 * denoised_cur + 0.5 * denoised_next
                h_i_eff = sigma_i.log() - sigma_next.log()
                x = (sigma_next / sigma_i) * x - torch.expm1(-h_i_eff) * D_corr
                denoised_prev = denoised_cur
                denoised_cur = denoised_next
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={"solver": "unipc_2pc_on_calibrated_seq_grid",
                      "schedule": "finite_N_calibrated_certificate_v4.1",
                      "num_steps": num_steps, "order": 2,
                      "step_sigmas": grid.tolist()},
        )
