"""AYS (Align Your Steps) schedule optimization on EDM CIFAR-10.

Reference: Sabour, Fidler, Kreis. "Align Your Steps: Optimizing
Sampling Schedules in Diffusion Models", ICML 2024.
research.nvidia.com/labs/toronto-ai/AlignYourSteps/

The AYS objective is the KL upper bound (KLUB) between the continuous
SDE reverse path and the discrete-sampler-induced path. In sigma
coordinate (EDM convention, sigma decreasing from sigma_max to 0) the
KLUB is

    KLUB(Pi_N) = Sum_{i=1}^{N}  int_{sigma_i}^{sigma_{i-1}}
                   (1 / sigma^3) * E_x || D_theta(x_sigma, sigma)
                                       - D_theta(x_{sigma_{i-1}},
                                                 sigma_{i-1}) ||^2 d sigma

where D_theta is the EDM denoiser, Pi_N = {sigma_0 > sigma_1 > ... > sigma_N}
is the candidate grid, and the expectation is over reverse-process
samples at level sigma (drawn from the Heun-Karras bootstrap, as in
proposed_control's calibration).

The 1/sigma^3 weight is the AYS terminal-sensitivity factor; it
amplifies errors at low sigma, where the data manifold is most rigid.

The schedule is found by minimising KLUB over the K-1 free interior
sigmas:

    Pi_{N}^{AYS} = argmin_{Pi_N} KLUB(Pi_N)

with monotone-grid constraints. The reference D_theta(x_sigma, sigma)
values are precomputed once on a fine log-sigma grid; for any candidate
Pi_N the KLUB integral is evaluated by trapezoidal rule on the fine
grid restricted to each interval. This avoids re-running the network
per optimizer step.
"""
from __future__ import annotations

from dataclasses import dataclass

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
# Precompute D_theta(x_sigma, sigma) on a fine log-sigma grid from the
# Heun-Karras bootstrap trajectory.
# ---------------------------------------------------------------------------

@dataclass
class DenoisedTable:
    """Bootstrap denoiser evaluations at each fine sigma."""
    sigma_grid: np.ndarray              # length M, descending
    denoised_per_sigma: torch.Tensor    # shape (M, B, C, H, W) on CPU
    meta: dict


@torch.inference_mode()
def precompute_denoised_table(
    net,
    *,
    num_calib_samples: int = 8,
    num_fine_sigmas: int = 128,
    sigma_min_clip: float = 0.002,
    sigma_max_clip: float = 80.0,
    seed: int = 0xA75,
    device: str | torch.device = "cuda",
    image_shape: tuple[int, int, int] | None = None,
) -> DenoisedTable:
    """Run a Heun-Karras bootstrap, then evaluate D_theta(x_sigma, sigma)
    at `num_fine_sigmas` log-uniform sigma values."""
    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    sigma_min_eff, sigma_max_eff = resolve_sigma_range(net)
    sigma_min_eff = max(sigma_min_eff, sigma_min_clip)
    sigma_max_eff = min(sigma_max_eff, sigma_max_clip)

    sigma_grid = np.exp(np.linspace(
        np.log(sigma_max_eff), np.log(sigma_min_eff), num_fine_sigmas
    )).astype(np.float32)

    # Bootstrap Heun-Karras trajectory at the fine sigma grid (treat sigma_grid
    # as the trajectory grid for simplicity; runs num_fine_sigmas-1 Heun steps).
    sigma_grid_t = torch.tensor(sigma_grid, dtype=torch.float32, device=device)
    x = sample_initial_noise((num_calib_samples, *shape), float(sigma_grid_t[0]),
                             seed=seed, device=device)
    denoised_per_sigma = []
    denoised_per_sigma.append(denoise(net, x, sigma_grid_t[0]).cpu())
    for k in range(num_fine_sigmas - 1):
        sa, sb = sigma_grid_t[k], sigma_grid_t[k + 1]
        denoised = denoise(net, x, sa)
        d = (x - denoised) / sa
        x_next = x + (sb - sa) * d
        if sb.item() > 0:
            denoised2 = denoise(net, x_next, sb)
            d2 = (x_next - denoised2) / sb
            x_next = x + (sb - sa) * 0.5 * (d + d2)
        x = x_next
        denoised_per_sigma.append(denoise(net, x, sb).cpu())

    denoised_tensor = torch.stack(denoised_per_sigma, dim=0)
    return DenoisedTable(
        sigma_grid=sigma_grid,
        denoised_per_sigma=denoised_tensor,
        meta={
            "num_calib_samples": num_calib_samples,
            "num_fine_sigmas": num_fine_sigmas,
            "sigma_min": sigma_min_eff,
            "sigma_max": sigma_max_eff,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# KLUB loss for a candidate grid Pi_N
# ---------------------------------------------------------------------------

def _D_at_sigma_via_interp(table: DenoisedTable, sigma_target: float) -> torch.Tensor:
    """Linear log-sigma interpolation of D_theta into table. Returns shape (B,C,H,W)."""
    grid = table.sigma_grid
    log_grid = np.log(grid)        # descending in i
    log_t = np.log(max(sigma_target, grid.min()))
    # log_grid is monotone-decreasing; find bracketing pair
    i = int(np.clip(np.searchsorted(-log_grid, -log_t) - 1, 0, len(log_grid) - 2))
    log_lo, log_hi = log_grid[i], log_grid[i + 1]    # log_lo > log_hi (since descending)
    if log_lo == log_hi:
        return table.denoised_per_sigma[i]
    f = (log_lo - log_t) / (log_lo - log_hi)
    f = float(np.clip(f, 0.0, 1.0))
    return (1 - f) * table.denoised_per_sigma[i] + f * table.denoised_per_sigma[i + 1]


def klub_loss(sigmas: np.ndarray, table: DenoisedTable) -> float:
    """KLUB(Pi_N) = sum_i integral over [sigma_i, sigma_{i-1}] of
        (1/sigma^3) * mean_B ||D(x_sigma, sigma) - D(x_{sigma_{i-1}}, sigma_{i-1})||^2
    where the inner expectation uses the bootstrap fine-grid evaluations.

    Trapezoidal integration on the subset of fine-grid points that fall
    inside each interval (plus interpolated endpoints).
    """
    grid = table.sigma_grid
    N_intervals = sigmas.shape[0] - 1
    sigma_floor = float(grid.min())
    total = 0.0
    for i in range(N_intervals):
        sigma_lo = float(sigmas[i + 1])           # right endpoint (smaller)
        sigma_hi = float(sigmas[i])               # left endpoint (larger)
        if sigma_hi <= sigma_floor:
            continue
        sigma_lo = max(sigma_lo, sigma_floor)
        if sigma_lo >= sigma_hi:
            continue
        D_ref = _D_at_sigma_via_interp(table, sigma_hi)         # holds at left endpoint
        # Fine-grid sigmas inside this interval
        in_interval = (grid >= sigma_lo) & (grid <= sigma_hi)
        sigma_inside = grid[in_interval]
        D_inside = table.denoised_per_sigma[in_interval]
        # Always include endpoints
        sigma_lo_arr = np.array([sigma_lo])
        sigma_hi_arr = np.array([sigma_hi])
        sigma_full = np.concatenate([sigma_hi_arr, sigma_inside[::-1] if sigma_inside.size else sigma_inside, sigma_lo_arr])
        D_lo = _D_at_sigma_via_interp(table, sigma_lo).unsqueeze(0)
        D_hi = D_ref.unsqueeze(0)
        if D_inside.shape[0] == 0:
            D_full = torch.cat([D_hi, D_lo], dim=0)
        else:
            D_full = torch.cat([D_hi, D_inside.flip(0), D_lo], dim=0)
        # Integrate (1/sigma^3) * mean_pixel_batch ||D_full[j] - D_ref||^2 d sigma
        # over sigma_full (descending from sigma_hi to sigma_lo).
        # Take sigma_full ascending for stable trapezoidal:
        order = np.argsort(sigma_full)
        sigma_sorted = sigma_full[order]
        D_sorted = D_full[order]
        # squared error per fine sigma, averaged over batch and pixels
        err_sq = (D_sorted - D_ref.unsqueeze(0)).pow(2).mean(dim=tuple(range(1, D_sorted.dim())))
        weight = 1.0 / (sigma_sorted ** 3 + 1e-12)
        integrand = err_sq.numpy() * weight
        # trapezoidal
        seg = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(sigma_sorted)
        total += float(seg.sum())
    return total


# ---------------------------------------------------------------------------
# Optimize the K-1 free interior sigmas
# ---------------------------------------------------------------------------

def optimal_ays_grid(table: DenoisedTable, K: int, *, init: str = "karras",
                     maxiter: int = 100) -> np.ndarray:
    """Return Pi_K = (sigma_max, K-1 interior, sigma_min, 0) minimising KLUB."""
    from scipy import optimize as _opt
    sigma_min = float(table.sigma_grid.min())
    sigma_max = float(table.sigma_grid.max())
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
        return klub_loss(params_to_sigmas(deltas), table)

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

    # Use a derivative-free optimizer since the loss is piecewise from
    # trapezoidal integration on a fine grid (gradients are noisy but
    # the loss surface is smooth in deltas at a coarser resolution).
    res = _opt.minimize(loss, deltas_init, method="Nelder-Mead",
                        options=dict(maxiter=maxiter * (K + 2), xatol=1e-3, fatol=1e-6,
                                     adaptive=True, disp=False))
    return params_to_sigmas(res.x).astype(np.float32)
