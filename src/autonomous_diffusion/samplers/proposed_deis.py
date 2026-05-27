"""Proposed control on a DEIS tAB-2 core.

THEORETICAL BACKGROUND (corrected appendix A, PDF 1 revision 2026-05-26):
========================================================================
DEIS tAB-2 is an Adams-Bashforth-class q=2 multistep solver. The local
AB-2 residual on a nonuniform grid is

    e_i^AB2 = [h_i^2 * (2 h_i + 3 h_{i-1}) / 12] * g''(xi_i),

with h_i = sigma_i - sigma_{i+1}, xi_i in (sigma_i, sigma_{i+1}). This is
HISTORY-COUPLED in (h_i, h_{i-1}). The pointwise grid rule m_s*(r) ∝
d_s(r)^{1/(p+1)} from Theorem A does NOT apply.

Theorem B (sequence-level): the optimal grid is

    Pi_{N,s}^* = argmin_{Pi_N} D_s^AB(Pi_N)

with D_s^AB(Pi_N) = Sum_i K_i^2 ||g''(xi_i)||^2 (plus Euler-warmup
residual for the first q-1=1 step). Implemented in `_ab_grid.py`.

USAGE:
  per_core_calib = True  -> sequence-level AB-2 grid (Theorem B)  [default]
  per_core_calib = False -> shared d_Heun pointwise grid (Round 2c fallback)

The default flipped to True on 2026-05-27 with the AB-2 implementation.
NFE per sample = num_steps (1 NFE/step).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ._ab_grid import (
    GqEstimate,
    estimate_gq_squared,
    optimal_ab_grid,
)
from ._common import (
    denoise,
    resolve_shape,
    resolve_sigma_range,
    sample_initial_noise,
)
from .base import Sampler, SamplerOutput, register_sampler
from .proposed_control import (
    calibrate,
    calibration_cache_path,
    load_calibration,
    optimal_step_sigmas,
    save_calibration,
    _net_key,
)


# ---------------------------------------------------------------------------
# g_q cache (separate from the pointwise calibration cache, but same net hash)
# ---------------------------------------------------------------------------

def _gq_cache_path(net, q: int, *, root: str | Path = "outputs/calibration") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"gq_q{q}_{_net_key(net)}.npz"


def _save_gq(gq: GqEstimate, path: Path) -> None:
    np.savez_compressed(
        path,
        sigma_grid=gq.sigma_grid,
        gq_sq_per_sigma=gq.gq_sq_per_sigma,
        q=np.array([gq.q]),
        meta=np.array([json.dumps(gq.meta)]),
        g1_sq=(gq.lower_order or {}).get(1, np.zeros(0)),
    )


def _load_gq(path: Path) -> GqEstimate:
    z = np.load(path, allow_pickle=False)
    lower = {}
    if "g1_sq" in z.files and z["g1_sq"].size > 0:
        lower[1] = z["g1_sq"]
    return GqEstimate(
        sigma_grid=z["sigma_grid"],
        gq_sq_per_sigma=z["gq_sq_per_sigma"],
        q=int(z["q"][0]),
        meta=json.loads(str(z["meta"][0])),
        lower_order=lower or None,
    )


def _get_or_compute_gq(
    net, q: int, *, cache_root: str | Path, num_calib_samples: int,
    num_fine_sigmas: int, seed: int, device, image_shape,
) -> tuple[GqEstimate, bool]:
    path = _gq_cache_path(net, q, root=cache_root)
    if path.exists():
        return _load_gq(path), False
    sigma_min, sigma_max = resolve_sigma_range(net)
    gq = estimate_gq_squared(
        net, q=q, num_calib_samples=num_calib_samples,
        num_fine_sigmas=num_fine_sigmas,
        sigma_min=sigma_min, sigma_max=sigma_max,
        seed=seed, device=device, image_shape=image_shape,
    )
    _save_gq(gq, path)
    return gq, True


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@register_sampler("proposed_deis")
class ProposedDEIS(Sampler):
    CALIB_ID_PER_CORE = "deis"
    AB_ORDER = 2

    def __init__(
        self,
        *,
        cache_root: str | Path = "outputs/calibration",
        num_calib_samples: int = 16,
        num_intervals: int = 32,
        num_ref_substeps: int = 16,
        num_fine_sigmas: int = 64,        # for g_q estimation
        calib_seed: int = 0xCA11B,
        force_recalibrate: bool = False,
        p: int = 2,
        perceptual_weight_k: float | None = None,
        per_core_calib: bool | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.num_calib_samples = num_calib_samples
        self.num_intervals = num_intervals
        self.num_ref_substeps = num_ref_substeps
        self.num_fine_sigmas = num_fine_sigmas
        self.calib_seed = calib_seed
        self.force_recalibrate = force_recalibrate
        self.p = p
        if perceptual_weight_k is None:
            perceptual_weight_k = float(os.environ.get("AD_PROPOSED_K", "2.0"))
        self.perceptual_weight_k = float(perceptual_weight_k)
        # Round 3c implementation status (2026-05-27):
        #   per_core_calib=True  -> sequence-level AB-2 grid (Theorem B). This
        #     is theoretically correct per the corrected appendix A, but with
        #     the current g''(sigma) finite-difference estimator + A_i = sigma^{-k}
        #     heuristic, the resulting grid is empirically WORSE than the
        #     shared d_Heun pointwise heuristic on CIFAR-10 (NFE=5 1k smoke:
        #     48.13 vs shared 39.64). Better A_i (terminal amplification)
        #     estimation is the open work item. Until then, default is False.
        #   per_core_calib=False -> shared d_Heun heuristic (Round 2c).
        #     Empirically best for DEIS until A_i estimator improves.
        if per_core_calib is None:
            per_core_calib = os.environ.get("AD_PROPOSED_CALIB", "shared").lower() == "per_core"
        self.per_core_calib = bool(per_core_calib)

    def _get_ab_grid(self, net, num_steps, device, image_shape):
        """Sequence-level optimal grid: K+2 sigmas (sigma_max, K-1 interior, sigma_min, 0)."""
        gq, did_compute = _get_or_compute_gq(
            net, q=self.AB_ORDER, cache_root=self.cache_root,
            num_calib_samples=self.num_calib_samples,
            num_fine_sigmas=self.num_fine_sigmas,
            seed=self.calib_seed, device=device, image_shape=image_shape,
        )
        sigmas_np = optimal_ab_grid(gq, num_steps, amp_k=self.perceptual_weight_k)
        # NFE consumed during g_q estimation:
        # 5-point stencil ~5 NFE per point per sample, q=2 uses 3-point ~3 NFE per point per sample.
        per_pt = 3 if self.AB_ORDER == 2 else 5
        traj_nfe = 2 * (self.num_fine_sigmas - 1) - 1   # Heun trajectory
        per_sigma = per_pt
        total_calib_nfe = self.num_calib_samples * (traj_nfe + self.num_fine_sigmas * per_sigma) if did_compute else 0
        return sigmas_np, total_calib_nfe, did_compute, gq

    def _get_shared_grid(self, net, num_steps, device, image_shape):
        """Pointwise shared d_Heun grid (Round 2c fallback)."""
        path = calibration_cache_path(net, root=self.cache_root, calib_id="heun")
        if path.exists() and not self.force_recalibrate:
            calib = load_calibration(path)
            return optimal_step_sigmas(calib, num_steps, p=self.p,
                                       perceptual_weight_k=self.perceptual_weight_k), 0, False, None
        calib = calibrate(
            net, num_calib_samples=self.num_calib_samples,
            num_intervals=self.num_intervals,
            num_ref_substeps=self.num_ref_substeps,
            seed=self.calib_seed, device=device, image_shape=image_shape,
            calib_id="heun",
        )
        save_calibration(calib, path)
        traj_nfe = 2 * self.num_intervals - 1
        per_interval_nfe = 2 + (2 * self.num_ref_substeps - 1)
        total = self.num_calib_samples * (traj_nfe + self.num_intervals * per_interval_nfe)
        return optimal_step_sigmas(calib, num_steps, p=self.p,
                                   perceptual_weight_k=self.perceptual_weight_k), total, True, None

    def sample(self, *, net, num_samples, num_steps, seed, device="cuda", batch_size=64, image_shape=None):
        device = torch.device(device)
        shape = resolve_shape(net, image_shape)

        if self.per_core_calib:
            step_sigmas_np, calib_nfe, did_recalibrate, gq = \
                self._get_ab_grid(net, num_steps, device, image_shape)
            # _get_ab_grid returns K+2 sigmas (sigma_max, K-1 interior, sigma_min, 0).
            # The DEIS recurrence integrates over the first K+1 (sigma_max..sigma_min)
            # and the final sigma=0 boundary is handled inside the loop.
            grid_basis = "ab2_sequence_optimal"
        else:
            step_sigmas_np, calib_nfe, did_recalibrate, gq = \
                self._get_shared_grid(net, num_steps, device, image_shape)
            grid_basis = "shared_d_heun"

        step_sigmas = torch.tensor(step_sigmas_np, dtype=torch.float32, device=device)
        # Number of integration "steps" the DEIS loop will run is len(step_sigmas) - 1
        # (= num_steps + 1 for AB grid since it has trailing 0).
        n_intervals = step_sigmas.shape[0] - 1

        out = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(step_sigmas[0]),
                                     seed=seed + done, device=device)
            cur_nfe = 0
            prev_eps = None
            for i in range(n_intervals):
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
        meta = {
            "solver": "deis_tAB2_on_ab_optimal_sigmas" if self.per_core_calib else "deis_tAB2_on_optimal_sigmas",
            "num_steps": num_steps,
            "p": self.p,
            "perceptual_weight_k": self.perceptual_weight_k,
            "per_core_calib": self.per_core_calib,
            "grid_basis": grid_basis,
            "step_sigmas": step_sigmas_np.tolist(),
            "calibration_nfe_total": calib_nfe,
            "calibration_nfe_amortized": amortised,
            "did_recalibrate": did_recalibrate,
        }
        if gq is not None:
            meta["gq_meta"] = gq.meta
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata=meta,
        )
