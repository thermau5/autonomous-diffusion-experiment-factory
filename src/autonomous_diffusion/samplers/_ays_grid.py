"""AYS-style (Align Your Steps) schedule optimization on EDM CIFAR-10.

STATUS (2026-05-27): EXPERIMENTAL / NEGATIVE RESULT.
The simplified pairwise-loss-table implementation in this file
UNDERPERFORMS the Karras rho=7 baseline at every tested K on 1k
CIFAR-10 smoke (UniPC core, single seed):

    K     this_AYS_FID    Karras_FID    (Ours, UniPC) m_s* FID (locked)
    5     74.42           65.22         21.46
    8     53.89           39.00          9.18
   12     44.12           32.74          5.58
   18     36.91           31.53          4.66

This is NOT a refutation of the AYS paper -- it is a refutation of
this specific pairwise-loss approximation. A faithful AYS reproduction
requires:
  - trajectory-level KL objective (not summed pairwise step errors);
  - differentiable backward-trajectory simulation;
  - the paper's specific Gaussian local-transition closed form.
That implementation is the ~1-2 day item this file does NOT achieve.

Kept as a research artifact and honest negative-result reference. Not
registered as a sampler.

Reference: Sabour, Hayat, Garg et al. "Align Your Steps: Optimizing
Sampling Schedules in Diffusion Models", ICLR 2024.

Original docstring follows.

This is NOT a faithful reproduction of the AYS paper (the original
optimizes a path-KL functional with closed-form Gaussian local
transitions tied to the SDE). It is the EDM-pretrained ODE analog:

  For each interval [sigma_a, sigma_b], compute the squared
  prediction-error of one solver step against a high-NFE reference
  trajectory, weighted by 1 / sigma_next^2 (terminal-sensitivity proxy):

    L_AYS(Pi_K)
    = Sum_i  ( 1 / sigma_{i+1}^2 ) *
            || step_solver(x_{Pi(sigma_i)}; sigma_i -> sigma_{i+1})
               - x^{ref}(sigma_{i+1}) ||^2,

  with x^{ref} drawn from the Heun-Karras-32 bootstrap trajectory used
  by proposed_control's calibration.

The schedule is then Pi_K^{AYS} = argmin_{Pi_K} L_AYS(Pi_K) under
monotone-grid constraints. Solved by L-BFGS-B with softmax(deltas)
parameterisation of K positive log-sigma gaps summing to log_range.

Calibration is cached per (net, target solver) so re-derivation is a
one-time cost. The result is a fixed schedule -- a learned scheduler in
the m_phi class (low-dimensional under-the-hood, since we optimize only
the K-1 free interior sigmas, but conceptually it's optimized end-to-end
against the path-KL objective rather than against a pointwise truncation
density).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ._common import (
    denoise,
    karras_sigmas,
    resolve_shape,
    resolve_sigma_range,
    sample_initial_noise,
)


# ---------------------------------------------------------------------------
# Tabulate per-interval AYS loss on a fine grid of (sigma_a, sigma_b) pairs
# ---------------------------------------------------------------------------

@dataclass
class AYSLossTable:
    """Pre-tabulated AYS per-step loss L(sigma_a, sigma_b)."""
    sigma_grid: np.ndarray            # length M, fine descending grid
    L_table: np.ndarray               # shape (M, M), L[i, j] for sigma_a=sigma_grid[i], sigma_b=sigma_grid[j]; j>i
    meta: dict


@torch.inference_mode()
def tabulate_ays_loss(
    net,
    target_step_fn,
    *,
    num_calib_samples: int = 16,
    num_fine_sigmas: int = 48,
    sigma_min_clip: float = 0.002,
    sigma_max_clip: float = 80.0,
    seed: int = 0xA75,
    device: str | torch.device = "cuda",
    image_shape: tuple[int, int, int] | None = None,
) -> AYSLossTable:
    """For each pair (i, j) of fine-grid sigmas with i < j (descending), compute
    the AYS per-step loss

      L(sigma_grid[i], sigma_grid[j]) =
          mean_B || target_step_fn(x_i; sigma_grid[i] -> sigma_grid[j])
                  - x_j^{ref} ||^2  /  sigma_grid[j]^2

    where x_i^{ref} are drawn from a Heun-Karras-32 bootstrap trajectory
    (the same bootstrap as proposed_control's calibration). The factor
    1/sigma_next^2 is the AYS terminal-sensitivity weight.

    target_step_fn(net, x, sigma_a, sigma_b, prev_history)
        -> (x_next, new_history)  is the solver's single-step update.
        For multistep solvers, prev_history is the previous denoised eval;
        we feed history from the bootstrap trajectory at the previous
        coarse-grid sigma.
    """
    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    sigma_min_eff, sigma_max_eff = resolve_sigma_range(net)
    sigma_min_eff = max(sigma_min_eff, sigma_min_clip)
    sigma_max_eff = min(sigma_max_eff, sigma_max_clip)

    # Descending fine grid in log sigma.
    log_lo, log_hi = np.log(sigma_min_eff), np.log(sigma_max_eff)
    sigma_grid = np.exp(np.linspace(log_hi, log_lo, num_fine_sigmas)).astype(np.float32)

    # Bootstrap reference trajectory on a Karras-32 grid (this is the same
    # bootstrap proposed_control uses).
    bootstrap_grid = karras_sigmas(num_fine_sigmas - 1, sigma_min_eff, sigma_max_eff,
                                    device=device).to(torch.float32)
    x = sample_initial_noise((num_calib_samples, *shape), float(bootstrap_grid[0]),
                             seed=seed, device=device)
    bootstrap_xs = [x]
    bootstrap_denoised = [denoise(net, x, bootstrap_grid[0])]
    for k in range(num_fine_sigmas - 1):
        sa, sb = bootstrap_grid[k], bootstrap_grid[k + 1]
        denoised = denoise(net, x, sa)
        d = (x - denoised) / sa
        x_next = x + (sb - sa) * d
        if sb.item() > 0:
            denoised2 = denoise(net, x_next, sb)
            d2 = (x_next - denoised2) / sb
            x_next = x + (sb - sa) * 0.5 * (d + d2)
        x = x_next
        bootstrap_xs.append(x)
        bootstrap_denoised.append(denoise(net, x, sb))

    bootstrap_sigmas_np = bootstrap_grid.cpu().numpy()

    def nearest_idx(sigma_target):
        return int(np.argmin(np.abs(bootstrap_sigmas_np - float(sigma_target))))

    # Tabulate L(i, j) for j > i.
    L_table = np.zeros((num_fine_sigmas, num_fine_sigmas), dtype=np.float64)
    L_table.fill(np.nan)

    for i in range(num_fine_sigmas - 1):
        sigma_a = float(sigma_grid[i])
        idx_a = nearest_idx(sigma_a)
        x_a = bootstrap_xs[idx_a]
        # For multistep history, pick the previous bootstrap denoised value.
        denoised_prev = bootstrap_denoised[max(idx_a - 1, 0)]
        sigma_prev = bootstrap_grid[max(idx_a - 1, 0)] if idx_a >= 1 else None
        for j in range(i + 1, num_fine_sigmas):
            sigma_b = float(sigma_grid[j])
            sigma_a_t = torch.tensor(sigma_a, device=device, dtype=torch.float32)
            sigma_b_t = torch.tensor(sigma_b, device=device, dtype=torch.float32)
            x_next, _ = target_step_fn(
                net, x_a, sigma_a_t, sigma_b_t,
                denoised_prev=denoised_prev, sigma_prev=sigma_prev,
            )
            # Reference at sigma_b: nearest bootstrap x.
            idx_b = nearest_idx(sigma_b)
            x_ref = bootstrap_xs[idx_b]
            err2 = (x_next - x_ref).pow(2).mean().item()
            L_table[i, j] = err2 / max(sigma_b ** 2, 1e-12)

    return AYSLossTable(
        sigma_grid=sigma_grid,
        L_table=L_table,
        meta={
            "num_calib_samples": num_calib_samples,
            "num_fine_sigmas": num_fine_sigmas,
            "sigma_min": sigma_min_eff,
            "sigma_max": sigma_max_eff,
            "seed": seed,
            "bootstrap": "heun_karras_32",
        },
    )


# Step-fn for UniPC (single step + 1-NFE corrector reuse)
def unipc_single_step_for_ays(net, x_cur, sigma_a, sigma_b,
                               denoised_prev=None, sigma_prev=None):
    denoised_cur = denoise(net, x_cur, sigma_a)
    if denoised_prev is None or sigma_prev is None:
        h_i = sigma_a.log() - sigma_b.log()
        x_pred = (sigma_b / sigma_a) * x_cur - torch.expm1(-h_i) * denoised_cur
    else:
        h_i_1 = sigma_prev.log() - sigma_a.log()
        h_i = sigma_a.log() - sigma_b.log()
        r = h_i_1 / h_i
        D_pred = (1 + 1 / (2 * r)) * denoised_cur - (1 / (2 * r)) * denoised_prev
        x_pred = (sigma_b / sigma_a) * x_cur - torch.expm1(-h_i) * D_pred
    denoised_next = denoise(net, x_pred, sigma_b)
    D_corr = 0.5 * denoised_cur + 0.5 * denoised_next
    h_i_eff = sigma_a.log() - sigma_b.log()
    x_next = (sigma_b / sigma_a) * x_cur - torch.expm1(-h_i_eff) * D_corr
    return x_next, denoised_cur


# ---------------------------------------------------------------------------
# Sequence optimization over Pi_K = (sigma_0, ..., sigma_K)
# ---------------------------------------------------------------------------

def _interp_L(ays: AYSLossTable, sigma_a: float, sigma_b: float) -> float:
    """Bilinear interpolation of L(sigma_a, sigma_b) on the (log-sigma_a,
    log-sigma_b) grid. The table only has entries for j > i (sigma_a >
    sigma_b on a descending grid); we constrain the lookup likewise.
    """
    grid = ays.sigma_grid
    if sigma_b >= sigma_a:
        return 0.0   # not a valid descending interval
    log_grid = np.log(grid)            # length M, monotone-decreasing in i
    log_a = np.log(max(sigma_a, grid.min()))
    log_b = np.log(max(sigma_b, grid.min()))
    # bracketing indices: find i_a such that log_grid[i_a] >= log_a >= log_grid[i_a+1]
    # since log_grid is descending, use searchsorted on the negated array
    i_a = int(np.clip(np.searchsorted(-log_grid, -log_a) - 1, 0, len(log_grid) - 2))
    i_b = int(np.clip(np.searchsorted(-log_grid, -log_b) - 1, 0, len(log_grid) - 2))
    # ensure i_b > i_a (since j > i in the table)
    if i_b <= i_a:
        i_b = min(i_a + 1, len(log_grid) - 1)
    # bilinear fractions in log-sigma
    log_a_lo, log_a_hi = log_grid[i_a], log_grid[i_a + 1]    # log_a_lo > log_a_hi
    log_b_lo, log_b_hi = log_grid[i_b], log_grid[min(i_b + 1, len(log_grid) - 1)]
    fa = (log_a_lo - log_a) / max(log_a_lo - log_a_hi, 1e-9)
    fa = float(np.clip(fa, 0.0, 1.0))
    fb = (log_b_lo - log_b) / max(log_b_lo - log_b_hi, 1e-9)
    fb = float(np.clip(fb, 0.0, 1.0))
    # corner values (clip i_b+1 to last)
    M = ays.L_table.shape[0]
    ib1 = min(i_b + 1, M - 1)
    # All four corners must satisfy j > i. If i_a + 1 >= i_b, we're at the
    # diagonal edge -- use the safe pair (i_a, i_b).
    if i_a + 1 >= i_b:
        return float(ays.L_table[i_a, i_b])
    L00 = ays.L_table[i_a,     i_b]
    L01 = ays.L_table[i_a,     ib1]
    L10 = ays.L_table[i_a + 1, i_b]
    L11 = ays.L_table[i_a + 1, ib1]
    if np.isnan(L00) or np.isnan(L01) or np.isnan(L10) or np.isnan(L11):
        return float(L00 if not np.isnan(L00) else ays.L_table[i_a, i_b])
    L_a_lo = (1 - fb) * L00 + fb * L01
    L_a_hi = (1 - fb) * L10 + fb * L11
    return float((1 - fa) * L_a_lo + fa * L_a_hi)


def ays_total_loss(sigmas: np.ndarray, ays: AYSLossTable) -> float:
    """Sum L over intervals of Pi_K = sigmas[0] > ... > sigmas[K] > 0."""
    K = sigmas.shape[0] - 1
    if K < 1:
        return 0.0
    total = 0.0
    sigma_floor = float(ays.sigma_grid.min())
    for i in range(K):
        sa, sb = float(sigmas[i]), float(sigmas[i + 1])
        if sb <= sigma_floor * 0.9999:
            continue
        total += _interp_L(ays, sa, sb)
    return total


def optimal_ays_grid(ays: AYSLossTable, K: int, *, init: str = "karras",
                     maxiter: int = 200) -> np.ndarray:
    """Find Pi_K of K+2 points (sigma_max, K-1 interior, sigma_min, 0) that
    minimize L_AYS via L-BFGS-B."""
    from scipy import optimize as _opt
    sigma_min = float(ays.sigma_grid.min())
    sigma_max = float(ays.sigma_grid.max())
    log_range = np.log(sigma_max) - np.log(sigma_min)

    def params_to_sigmas(deltas: np.ndarray) -> np.ndarray:
        d = deltas - np.max(deltas)
        ex = np.exp(d)
        gaps = ex / ex.sum()
        log_sigmas_inner = np.log(sigma_max) - np.cumsum(gaps) * log_range
        sigmas = np.empty(K + 2, dtype=np.float64)
        sigmas[0] = sigma_max
        sigmas[1:K + 1] = np.exp(log_sigmas_inner)
        sigmas[K + 1] = 0.0
        return sigmas

    def loss(deltas):
        return ays_total_loss(params_to_sigmas(deltas), ays)

    if init == "karras":
        with torch.no_grad():
            kar = karras_sigmas(K, sigma_min, sigma_max, device="cpu").cpu().numpy()
        kar_inner = kar[:K + 1] if kar[-1] == 0 else kar
        if kar_inner.shape[0] != K + 1:
            kar_inner = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), K + 1))
        log_gaps = -np.diff(np.log(np.clip(kar_inner, 1e-9, None))) / log_range
        gaps_norm = np.clip(log_gaps, 1e-9, None)
        gaps_norm = gaps_norm / gaps_norm.sum()
        deltas_init = np.log(gaps_norm + 1e-9)
        deltas_init -= deltas_init.mean()
    else:
        deltas_init = np.zeros(K, dtype=np.float64)

    res = _opt.minimize(loss, deltas_init, method="L-BFGS-B",
                        options=dict(maxiter=maxiter, disp=False))
    return params_to_sigmas(res.x).astype(np.float32)
