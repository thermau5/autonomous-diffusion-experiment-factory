"""Proposed control on a DEIS tAB-2 core.

NFE = num_steps (1 NFE/step).

THEORETICAL NOTE (corrected appendix A, 2026-05-26):
====================================================
DEIS tAB-2 is an Adams-Bashforth-class multistep solver. Per the corrected
PDF 1 appendix, the local AB residual is the interpolation-kernel residual

    e_i^AB2 = [h_i^2 (2 h_i + 3 h_{i-1}) / 12] * g''(xi_i),

which is HISTORY-COUPLED in (h_i, h_{i-1}). The pointwise grid rule
m_s*(r) ∝ d_s(r)^{1/(p+1)} from the single-step certificate is NOT a
theorem for AB-class solvers. The Round 3b ablation confirmed this
empirically: per-core d_DEIS estimated from one-step DEIS-vs-Heun-substep
error puts mass at mid-sigma (where the AB kernel coefficients dominate)
rather than at low-sigma (where g'' actually peaks for FID-sensitive
detail).

The corrected certificate is a SEQUENCE-LEVEL program:

    Pi_{N,s}^* = argmin_{Pi_N} D_s^AB(Pi_N),  D_s^AB = Sum_i A_i^2 |K_{s,i}^q g|^2

with monotone-grid constraints. Implementing this is Round 3c.

Until Round 3c is implemented, this sampler defaults to per_core_calib=False
i.e. uses the shared d_Heun grid. The Round 3b ablation showed shared-d
DEIS still wins at low NFE vs. baseline DEIS by reintroducing low-sigma
sensitivity, but this is not a per-core DEIS certificate -- it's a working
suboptimal grid.
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


@register_sampler("proposed_deis")
class ProposedDEIS(Sampler):
    CALIB_ID_PER_CORE = "deis"

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
        # Per-core calibration LOSES catastrophically on DEIS in the Round 3b
        # ablation: +25.5 / +13.2 / +7.2 / +3.7 at NFE 5/8/12/18. The
        # extrapolation-kernel-dominated single-step error doesn't track FID
        # sensitivity. Default OFF (use shared Heun calibration).
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
            prev_eps = None
            for i in range(num_steps):
                sigma_i = step_sigmas[i]
                sigma_next = step_sigmas[i + 1]
                denoised = denoise(net, x, sigma_i)
                cur_nfe += 1
                eps_i = (x - denoised) / sigma_i

                if sigma_next.item() == 0:
                    x = denoised
                    break

                if prev_eps is None:
                    x = x + (sigma_next - sigma_i) * eps_i
                else:
                    sigma_prev = step_sigmas[i - 1]
                    lam_i = -sigma_i.log()
                    lam_next = -sigma_next.log()
                    lam_prev = -sigma_prev.log()
                    h_i = lam_next - lam_i
                    h_prev = lam_i - lam_prev
                    coef_cur = 1 + h_i / (2 * h_prev)
                    coef_prev = -h_i / (2 * h_prev)
                    eps_extrap = coef_cur * eps_i + coef_prev * prev_eps
                    x = x + (sigma_next - sigma_i) * eps_extrap
                prev_eps = eps_i
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        amortised = calib_nfe / max(num_samples, 1)
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "deis_tAB2_on_optimal_sigmas",
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
