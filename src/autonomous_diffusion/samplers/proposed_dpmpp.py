"""Proposed control on a DPM-Solver++ 2M multistep core.

Reuses the *same* certificate calibration as `proposed_control` (the d(sigma)
estimator is solver-agnostic at 2nd order: the leading prefactor of the
local truncation error differs between Heun and DPM-Solver++, but the
*shape* d(sigma) is set by the integrand's 3rd derivative, which is a
property of the ODE not the solver). The step density formula is identical:

    m*(sigma) propto (d(sigma) * w(sigma))^{1/(p+1)},   p = 2, w = sigma^{-k}.

The only difference vs `proposed_control` is that the integration is done
with the DPM-Solver++ 2M multistep update instead of EDM-Heun. This buys
the 1-NFE-per-step economy of multistep solvers while keeping the
certificate's optimal step placement.

NFE = num_steps (multistep is 1 NFE per step).
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


def _dpmpp_step(net, x, sigma_i, sigma_next, sigma_prev=None, denoised_prev=None):
    """One DPM-Solver++ 2M step. Returns (x_next, denoised_at_sigma_i, nfe=1)."""
    denoised = denoise(net, x, sigma_i)
    if sigma_next.item() == 0 or denoised_prev is None:
        if sigma_next.item() == 0:
            return denoised, denoised, 1
        h_i = sigma_i.log() - sigma_next.log()
        x_next = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * denoised
        return x_next, denoised, 1
    h_i_1 = sigma_prev.log() - sigma_i.log()
    h_i = sigma_i.log() - sigma_next.log()
    r = h_i_1 / h_i
    D_i = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * denoised_prev
    x_next = (sigma_next / sigma_i) * x - torch.expm1(-h_i) * D_i
    return x_next, denoised, 1


@register_sampler("proposed_dpmpp")
class ProposedDPMpp(Sampler):
    """Certificate-optimal step density on a DPM-Solver++ 2M core.

    Shares the per-net calibration cache with `proposed_control`: the
    calibration depends on the underlying ODE (the EDM pretrained net),
    not on the integrator class.
    """
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

    def _get_calibration(self, net, device, image_shape):
        path = calibration_cache_path(net, root=self.cache_root)
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
            denoised_prev = None
            for i in range(num_steps):
                sigma_i = step_sigmas[i]
                sigma_next = step_sigmas[i + 1]
                sigma_prev = step_sigmas[i - 1] if i > 0 else None
                x, denoised_cur, used = _dpmpp_step(
                    net, x, sigma_i, sigma_next,
                    sigma_prev=sigma_prev, denoised_prev=denoised_prev,
                )
                denoised_prev = denoised_cur
                cur_nfe += used
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        amortised = calib_nfe / max(num_samples, 1)
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "dpm_solver_pp_2M_on_optimal_sigmas",
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
