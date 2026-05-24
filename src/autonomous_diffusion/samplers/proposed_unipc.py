"""Proposed control on a UniPC 2nd-order predictor-corrector core.

Same shared calibration as `proposed_control`/`proposed_dpmpp` (solver-
agnostic d(sigma) at 2nd order). The integration uses UniPC's predictor
(DPM-Solver++ 2M style with the previous denoised) plus a free corrector
on the new sigma (the next predictor step's denoised eval is reused as
the corrector, so net NFE = num_steps).
"""
from __future__ import annotations

from pathlib import Path

import torch

from ._common import (
    denoise,
    resolve_shape,
    sample_initial_noise,
)
from .base import Sampler, SamplerOutput, register_sampler
from .proposed_control import (
    calibrate,
    calibration_cache_path,
    load_calibration,
    optimal_step_sigmas,
    save_calibration,
)


@register_sampler("proposed_unipc")
class ProposedUniPC(Sampler):
    CALIB_ID_PER_CORE = "unipc"

    def __init__(
        self,
        *,
        cache_root: str | Path = "outputs/calibration",
        num_calib_samples: int = 16,
        num_intervals: int = 32,
        num_ref_substeps: int = 16,
        calib_seed: int = 0xCA11B,
        force_recalibrate: bool = False,
        p: int = 2,
        perceptual_weight_k: float | None = None,
        per_core_calib: bool | None = None,
    ):
        import os
        self.cache_root = Path(cache_root)
        self.num_calib_samples = num_calib_samples
        self.num_intervals = num_intervals
        self.num_ref_substeps = num_ref_substeps
        self.calib_seed = calib_seed
        self.force_recalibrate = force_recalibrate
        self.p = p
        if perceptual_weight_k is None:
            perceptual_weight_k = float(os.environ.get("AD_PROPOSED_K", "2.0"))
        self.perceptual_weight_k = float(perceptual_weight_k)
        if per_core_calib is None:
            per_core_calib = os.environ.get("AD_PROPOSED_CALIB", "shared").lower() == "per_core"
        self.per_core_calib = bool(per_core_calib)

    def _get_calibration(self, net, device, image_shape):
        calib_id = self.CALIB_ID_PER_CORE if self.per_core_calib else "heun"
        path = calibration_cache_path(net, root=self.cache_root, calib_id=calib_id)
        if path.exists() and not self.force_recalibrate:
            return load_calibration(path), 0, False
        calib = calibrate(
            net,
            num_calib_samples=self.num_calib_samples,
            num_intervals=self.num_intervals,
            num_ref_substeps=self.num_ref_substeps,
            seed=self.calib_seed,
            device=device,
            image_shape=image_shape,
            calib_id=calib_id,
        )
        save_calibration(calib, path)
        traj_nfe = 2 * self.num_intervals - 1
        per_interval_nfe = 2 + (2 * self.num_ref_substeps - 1)
        total_calib_nfe = self.num_calib_samples * (traj_nfe + self.num_intervals * per_interval_nfe)
        return calib, total_calib_nfe, True

    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        shape = resolve_shape(net, image_shape)
        calib, calib_nfe, did_recalibrate = self._get_calibration(net, device, image_shape)
        step_sigmas_np = optimal_step_sigmas(
            calib, num_steps, p=self.p, perceptual_weight_k=self.perceptual_weight_k,
        )
        step_sigmas = torch.tensor(step_sigmas_np, dtype=torch.float32, device=device)

        out = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(step_sigmas[0]),
                                     seed=seed + done, device=device)
            cur_nfe = 0
            # Initial denoised at sigma_0
            denoised_cur = denoise(net, x, step_sigmas[0])
            cur_nfe += 1
            denoised_prev = None

            for i in range(num_steps):
                sigma_i = step_sigmas[i]
                sigma_next = step_sigmas[i + 1]
                if sigma_next.item() == 0:
                    x = denoised_cur
                    break
                # Predictor (1st-order start, 2nd-order multistep otherwise)
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
                # Corrector: evaluate at predicted sigma_next; reuse as next iter's denoised_cur
                denoised_next = denoise(net, x_pred, sigma_next)
                cur_nfe += 1
                D_corr = 0.5 * denoised_cur + 0.5 * denoised_next
                h_i_eff = sigma_i.log() - sigma_next.log()
                x = (sigma_next / sigma_i) * x - torch.expm1(-h_i_eff) * D_corr
                denoised_prev = denoised_cur
                denoised_cur = denoised_next

            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        amortised = calib_nfe / max(num_samples, 1)
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "unipc_2pc_on_optimal_sigmas",
                "num_steps": num_steps,
                "p": self.p,
                "perceptual_weight_k": self.perceptual_weight_k,
                "step_sigmas": step_sigmas_np.tolist(),
                "calibration_nfe_total": calib_nfe,
                "calibration_nfe_amortized": amortised,
                "did_recalibrate": did_recalibrate,
                "calib_meta": calib.meta,
            },
        )
