"""Proposed control on a DPM-Solver-v3 core.

This applies our certificate-optimal grid `m_s*` on top of the same
EMS-corrected update law that DPM-Solver-v3 uses. The EMS coefficients
(l, s, b) come from the paper team's precomputed statistics for
`edm-cifar10-32x32-uncond-vp`; the schedule is replaced by our
calibrated `m_s*` (defaults to shared d_Heun calibration at order p=3
since DPM-Solver-v3 is order-3 by default).

This is the Level-2 comparison on the new Level-1 core: at fixed
EMS-corrected update law, does our schedule win over the paper-default
logSNR-uniform schedule?

The override mechanism: construct `DPM_Solver_v3` with skip_type="logSNR"
(any initialization is fine), then patch `.timesteps` and `.indexes` with
our `m_s*`-derived values via `convert_to_indexes` / `convert_to_timesteps`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from ._common import resolve_shape, resolve_sigma_range
from .base import Sampler, SamplerOutput, register_sampler
from .dpm_solver_v3 import DPMV3_DIR, _ensure_path, _select_stats_dir
from .proposed_control import (
    calibrate,
    calibration_cache_path,
    load_calibration,
    optimal_step_sigmas,
    save_calibration,
)


@register_sampler("proposed_dpm_solver_v3")
class ProposedDPMSolverV3(Sampler):
    CALIB_ID_PER_CORE = "dpm_solver_v3"

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
        order: int = 3,
        degenerated: bool = False,
    ):
        self.cache_root = Path(cache_root)
        self.num_calib_samples = num_calib_samples
        self.num_intervals = num_intervals
        self.num_ref_substeps = num_ref_substeps
        self.calib_seed = calib_seed
        self.force_recalibrate = force_recalibrate
        self.p = int(p)
        if perceptual_weight_k is None:
            perceptual_weight_k = float(os.environ.get("AD_PROPOSED_K", "2.0"))
        self.perceptual_weight_k = float(perceptual_weight_k)
        if per_core_calib is None:
            # Default to shared d_Heun until we have evidence per-core wins
            # on the v3 core; per-core was a Round-3b finding for UniPC/DPM++.
            per_core_calib = os.environ.get("AD_PROPOSED_CALIB", "shared").lower() == "per_core"
        self.per_core_calib = bool(per_core_calib)
        self.order = int(order)
        self.degenerated = bool(degenerated)

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

    def sample(self, *, net, num_samples, num_steps, seed,
               device="cuda", batch_size=64, image_shape=None):
        _ensure_path()
        from samplers.dpm_solver_v3 import DPM_Solver_v3
        from samplers.utils import NoiseScheduleEDM, model_wrapper

        device = torch.device(device)
        shape = resolve_shape(net, image_shape)
        sigma_min, sigma_max = resolve_sigma_range(net)

        # Build the optimal step sigmas from our certificate
        calib, calib_nfe, did_recal = self._get_calibration(net, device, image_shape)
        step_sigmas_np = optimal_step_sigmas(
            calib, num_steps, p=self.p,
            perceptual_weight_k=self.perceptual_weight_k,
        )
        # optimal_step_sigmas returns num_steps+1 values [sigma_max, ..., 0],
        # with the trailing 0 representing UniPC's boundary jump (sigma_min -> 0,
        # no NFE). DPM-Solver-v3 has no such free boundary step: it integrates
        # over [sigma_max, sigma_min] in num_steps sub-intervals (all NFE-paying).
        # Replace the trailing 0 with sigma_min so the closed-interval grid is
        # [sigma_max, sigma_{1}, ..., sigma_{num_steps-1}, sigma_min].
        adapted = np.array(step_sigmas_np, dtype=np.float64, copy=True)
        if adapted[-1] == 0:
            adapted[-1] = float(sigma_min)
        # Clamp to the EMS table's sigma range to avoid out-of-bounds indices.
        adapted = np.clip(adapted, float(sigma_min), float(sigma_max))
        # Enforce strict monotonic descent (clamping can create duplicates).
        for i in range(1, len(adapted)):
            if adapted[i] >= adapted[i - 1]:
                adapted[i] = adapted[i - 1] * 0.999
        custom_timesteps = torch.tensor(adapted, dtype=torch.float64, device=device)
        if custom_timesteps.shape[0] != num_steps + 1:
            raise RuntimeError(
                f"optimal_step_sigmas returned {custom_timesteps.shape[0]} values, "
                f"expected num_steps+1={num_steps + 1}."
            )

        # Construct the v3 solver with any default skip_type (we override).
        stats_dir = _select_stats_dir(num_steps)
        if not stats_dir.is_dir():
            raise FileNotFoundError(f"DPM-Solver-v3 EMS stats not found at {stats_dir}.")
        ns = NoiseScheduleEDM()
        dpm_v3 = DPM_Solver_v3(
            str(stats_dir), ns,
            steps=num_steps,
            t_start=float(sigma_max), t_end=float(sigma_min),
            skip_type="logSNR",
            device=str(device),
            degenerated=self.degenerated,
        )

        # Override the timestep grid with our m_s*.
        indexes = dpm_v3.convert_to_indexes(custom_timesteps)
        dpm_v3.indexes = indexes
        dpm_v3.timesteps = dpm_v3.convert_to_timesteps(indexes, str(device))

        def edm_x0_model(x, t_input, cond=None):
            if t_input.dim() == 0:
                t_b = t_input.expand(x.shape[0]).to(x.dtype)
            else:
                t_b = t_input.to(x.dtype)
            return net(x, t_b).to(x.dtype)
        model_fn = model_wrapper(edm_x0_model, ns, class_labels=None)

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            g = torch.Generator(device=device).manual_seed(int(seed + done))
            x_init = torch.randn(
                (b, *shape), generator=g, device=device, dtype=torch.float64
            ) * float(sigma_max)
            with torch.no_grad():
                x = dpm_v3.sample(
                    model_fn, x_init,
                    order=self.order,
                    p_pseudo=(num_steps <= 5),
                    use_corrector=(num_steps <= 6),
                    c_pseudo=False,
                    lower_order_final=True,
                )
            out.append(x.clamp(-1, 1).float().cpu())
            if done == 0:
                nfe_per_sample = num_steps
            done += b

        amortised = calib_nfe / max(num_samples, 1)
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "dpm_solver_v3_on_optimal_sigmas",
                "schedule": "m_star_s",
                "num_steps": num_steps,
                "p": self.p,
                "perceptual_weight_k": self.perceptual_weight_k,
                "ems_stats_dir": stats_dir.name,
                "step_sigmas": step_sigmas_np.tolist(),
                "calibration_nfe_total": calib_nfe,
                "calibration_nfe_amortized": amortised,
                "did_recalibrate": did_recal,
                "calib_meta": calib.meta,
            },
        )
