"""Proposed risk-constrained control scheduler.

Theory (Autonomous Diffusion PDFs, KL/NLL certificate):
    Q(u) = Q_0 + (c_stat / n) * \\int a(r) / rho(r) dr
                + c_disc      * \\int d(r) / m(r)^p   dr

The generator (pretrained EDM net) is frozen, so rho is not tunable. The
sampling-step density m is the only knob the proposed method optimizes
during validation. The certificate's stationary point for fixed total step
budget K is

    m*(r) = arg min_{\\int m = 1} \\int d(r) / m(r)^p dr
          \\propto d(r)^{1 / (p+1)}.

For an order-p solver (Heun has p = 2), this is the optimal step density.
We discretize: pick K+1 sigma values (sigma_0 = sigma_max, ..., sigma_K = 0)
that equalize the cumulative weight g(sigma) = d(sigma)^{1/(p+1)} ds across
intervals, i.e. inverse-CDF placement.

d(r) is estimated empirically on a small VALIDATION batch by comparing a
single Heun step against a fine reference. This is the only knob tuned
during validation; once chosen, the per-(net, sigma_min, sigma_max) sigma
grid is locked.

Calibration is cached on disk so the locked-test runner reuses exactly the
validation-time sigma grid without recomputing on test data.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._common import (
    denoise,
    karras_sigmas,
    resolve_shape,
    resolve_sigma_range,
    sample_initial_noise,
)
from .base import Sampler, SamplerOutput, register_sampler


# ---------------------------------------------------------------------------
# Calibration: estimate d(sigma) per interval on a fine Karras grid.
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    sigma_grid: np.ndarray         # length M+1, descending, sigma_max ... 0
    d_per_interval: np.ndarray     # length M, per-interval discretization weight
    p: int                         # solver order (Heun: 2)
    meta: dict


def _heun_step(net, x, sigma, sigma_next):
    denoised = denoise(net, x, sigma)
    d = (x - denoised) / sigma
    x_next = x + (sigma_next - sigma) * d
    if sigma_next.item() > 0:
        denoised2 = denoise(net, x_next, sigma_next)
        d2 = (x_next - denoised2) / sigma_next
        x_next = x + (sigma_next - sigma) * 0.5 * (d + d2)
    return x_next


def _heun_substeps(net, x, sigma, sigma_next, num_sub):
    sub = torch.linspace(float(sigma), float(sigma_next), num_sub + 1,
                         device=x.device, dtype=torch.float32)
    for k in range(num_sub):
        x = _heun_step(net, x, sub[k], sub[k + 1])
    return x


@torch.inference_mode()
def calibrate(
    net,
    *,
    num_calib_samples: int = 16,
    num_intervals: int = 32,
    num_ref_substeps: int = 16,
    seed: int = 0xCA11B,
    device: str | torch.device = "cuda",
    image_shape: tuple[int, int, int] | None = None,
    grid: str = "uniform_log",
) -> Calibration:
    """Empirically estimate d(sigma), the per-interval discretization weight.

    For each interval [sigma_a, sigma_b] on a (uniform-log by default) grid:
      single = Heun(x_a, sigma_a -> sigma_b, 1 step)
      ref    = Heun(x_a, sigma_a -> sigma_b, num_ref_substeps small steps)
      d_i    = mean ||single - ref|| / |sigma_a - sigma_b|   (per-unit-sigma)

    The intermediate x_a are sampled by integrating the same Karras trajectory
    (validation-only) so d(sigma) reflects errors on in-distribution noisy
    images, not on arbitrary noise.

    The final interval (sigma -> 0) is excluded from the d estimate because
    the reference substepping is numerically singular at sigma = 0; we copy
    the previous interval's d into that bin so optimal_step_sigmas places a
    sensible number of steps near zero without being dominated by noise.

    grid: 'uniform_log' (default, unbiased) or 'karras' (Karras-spaced).
    """
    device = torch.device(device)
    sigma_min, sigma_max = resolve_sigma_range(net)
    shape = resolve_shape(net, image_shape)
    if grid == "uniform_log":
        log_sigmas = torch.linspace(
            float(np.log(sigma_max)), float(np.log(sigma_min)),
            num_intervals + 1, dtype=torch.float64, device=device,
        )
        sigma_grid = torch.cat([torch.exp(log_sigmas), log_sigmas.new_zeros([1])]).to(torch.float32)
        # the final 0 is for the boundary-only step that we *exclude* below
    elif grid == "karras":
        sigma_grid = karras_sigmas(num_intervals, sigma_min, sigma_max, device=device).to(torch.float32)
    else:
        raise ValueError(f"unknown calibration grid {grid!r}")

    # Generate intermediate x_a values along a Karras trajectory (the canonical
    # validation-time distribution of noisy images at each sigma).
    karras_traj_sigmas = karras_sigmas(num_intervals, sigma_min, sigma_max, device=device).to(torch.float32)
    x_traj = sample_initial_noise((num_calib_samples, *shape), float(karras_traj_sigmas[0]),
                                  seed=seed, device=device)
    karras_xs = [x_traj]
    for i in range(num_intervals):
        x_traj = _heun_step(net, x_traj, karras_traj_sigmas[i], karras_traj_sigmas[i + 1])
        karras_xs.append(x_traj)

    def x_at(sigma_target):
        # nearest-sigma lookup into the Karras trajectory
        idx = int(np.argmin(np.abs(karras_traj_sigmas.cpu().numpy() - float(sigma_target))))
        return karras_xs[idx]

    # Compute d on each non-boundary interval of sigma_grid.
    M = sigma_grid.shape[0] - 2   # exclude trailing 0
    d_per_interval = np.zeros(M + 1, dtype=np.float64)
    for i in range(M):
        sigma_a = sigma_grid[i]
        sigma_b = sigma_grid[i + 1]
        x = x_at(sigma_a)
        single = _heun_step(net, x, sigma_a, sigma_b)
        ref = _heun_substeps(net, x, sigma_a, sigma_b, num_ref_substeps)
        diff = (single - ref).pow(2).mean(dim=list(range(1, single.dim()))).sqrt()
        interval_len = float(sigma_a - sigma_b).__abs__()
        d_per_interval[i] = float(diff.mean().item()) / max(interval_len, 1e-12)
    # Boundary interval to sigma=0: copy the previous bin's value (extrapolate)
    d_per_interval[M] = d_per_interval[M - 1] if M > 0 else 1.0

    return Calibration(
        sigma_grid=sigma_grid.cpu().numpy(),
        d_per_interval=d_per_interval,
        p=2,
        meta={
            "num_calib_samples": num_calib_samples,
            "num_intervals": num_intervals,
            "num_ref_substeps": num_ref_substeps,
            "seed": seed,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "grid": grid,
            "boundary_handling": "copy_previous",
        },
    )


# ---------------------------------------------------------------------------
# Inverse-CDF step placement from calibration.
# ---------------------------------------------------------------------------

def optimal_step_sigmas(
    calib: Calibration,
    num_steps: int,
    p: int | None = None,
    eps: float = 1e-12,
    perceptual_weight_k: float = 2.0,
) -> np.ndarray:
    """Pick num_steps+1 sigma values that equalize the cumulative weight
    g(sigma) = (d(sigma) * w(sigma))^{1/(p+1)} dsigma across intervals.

    `perceptual_weight_k` adds a perceptual weighting w(sigma) = sigma^{-k}
    to capture the fact that pixel error at low noise levels is more visible
    in the final image than the same pixel error at high noise levels. The
    raw certificate uses k=0 (pure pixel error); Karras's rho=7 schedule
    corresponds empirically to a much stronger low-sigma concentration than
    k=0 produces from empirical d. Default k=2 matches Karras's qualitative
    concentration; tune on validation.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    p = int(p if p is not None else calib.p)
    sigmas = calib.sigma_grid                  # length M+1 descending
    d = calib.d_per_interval                   # length M
    interval_mid = 0.5 * (sigmas[:-1] + sigmas[1:])
    interval_mid = np.clip(interval_mid, eps, None)
    intervals = np.abs(sigmas[:-1] - sigmas[1:])
    w = interval_mid ** (-perceptual_weight_k)
    weights = (np.clip(d, eps, None) * w) ** (1.0 / (p + 1)) * intervals
    cum = np.concatenate([[0.0], np.cumsum(weights)])
    target = np.linspace(0.0, cum[-1], num_steps + 1)
    new_sigmas = np.interp(target, cum, sigmas)
    new_sigmas[0] = sigmas[0]
    new_sigmas[-1] = 0.0
    return new_sigmas.astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration cache on disk
# ---------------------------------------------------------------------------

def _net_key(net) -> str:
    """Stable hash of the net's identity for cache lookup. Uses a small set
    of pretrained-net attributes that uniquely identify the checkpoint."""
    bits = []
    for k in ("img_resolution", "img_channels", "sigma_min", "sigma_max",
              "sigma_data", "label_dim"):
        if hasattr(net, k):
            bits.append(f"{k}={getattr(net, k)}")
    # Add a few-parameter hash so two structurally identical nets with
    # different weights don't collide.
    try:
        ps = list(net.parameters())
        sample = torch.cat([p.reshape(-1)[:32].detach().cpu().float().reshape(-1) for p in ps[:4]])
        bits.append("phash=" + hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16])
    except Exception:
        pass
    return hashlib.sha256("|".join(bits).encode()).hexdigest()[:16]


def calibration_cache_path(net, *, root: str | Path = "outputs/calibration") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"calib_{_net_key(net)}.npz"


def save_calibration(calib: Calibration, path: str | Path) -> None:
    np.savez_compressed(
        path,
        sigma_grid=calib.sigma_grid,
        d_per_interval=calib.d_per_interval,
        p=np.array([calib.p]),
        meta=np.array([json.dumps(calib.meta)]),
    )


def load_calibration(path: str | Path) -> Calibration:
    z = np.load(path, allow_pickle=False)
    return Calibration(
        sigma_grid=z["sigma_grid"],
        d_per_interval=z["d_per_interval"],
        p=int(z["p"][0]),
        meta=json.loads(str(z["meta"][0])),
    )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@register_sampler("proposed_control")
class ProposedControl(Sampler):
    """EDM-Heun integration on step sigmas derived from the per-(net,sigma_range)
    calibration. The calibration is computed once (validation) and cached.

    Calibration cost is amortized over all samples drawn with this net and
    reported in `metadata['calibration_nfe']` for honest accounting.
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
        self.cache_root = Path(cache_root)
        self.num_calib_samples = num_calib_samples
        self.num_intervals = num_intervals
        self.num_ref_substeps = num_ref_substeps
        self.calib_seed = calib_seed
        self.force_recalibrate = force_recalibrate
        self.p = p
        # Allow env override for sweep-time tuning without modifying call sites.
        if perceptual_weight_k is None:
            perceptual_weight_k = float(os.environ.get("AD_PROPOSED_K", "2.0"))
        self.perceptual_weight_k = float(perceptual_weight_k)

    def _get_calibration(self, net, device, image_shape) -> tuple[Calibration, int, bool]:
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
        # NFE consumed during calibration:
        #   1 trajectory of num_intervals Heun steps  -> 2*num_intervals - 1 NFE
        #   per interval: 1 single Heun + num_ref_substeps Heun  -> (2 + 2*num_ref_substeps - 1)
        traj_nfe = 2 * self.num_intervals - 1
        per_interval_nfe = (2) + (2 * self.num_ref_substeps - 1)
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

        out: list[torch.Tensor] = []
        nfe_per_sample = 0
        done = 0
        while done < num_samples:
            b = min(batch_size, num_samples - done)
            x = sample_initial_noise((b, *shape), float(step_sigmas[0]),
                                     seed=seed + done, device=device)
            cur_nfe = 0
            for i in range(num_steps):
                x = _heun_step(net, x, step_sigmas[i], step_sigmas[i + 1])
                cur_nfe += 2 if step_sigmas[i + 1].item() > 0 else 1
            out.append(x.clamp(-1, 1).cpu())
            if done == 0:
                nfe_per_sample = cur_nfe
            done += b

        amortised = calib_nfe / max(num_samples, 1)
        return SamplerOutput(
            samples=torch.cat(out, dim=0)[:num_samples],
            nfe=nfe_per_sample,
            metadata={
                "solver": "edm_heun_on_optimal_sigmas",
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
