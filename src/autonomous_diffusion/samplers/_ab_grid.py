"""Sequence-level optimal grid for q-step Adams-Bashforth solvers.

Per the corrected appendix A of "Autonomous Diffusion - Autonomous Diffusion.pdf"
(2026-05-26), the local truncation error of a q-step AB method on a
nonuniform grid is

    e_i^AB = (1 / q!) * g^{(q)}(xi_i) * int_{sigma_i}^{sigma_{i+1}}
             prod_{ell=0}^{q-1} (sigma - sigma_{i-ell}) d sigma.

For DEIS tAB-2 (q = 2):

    e_i^AB2 = [h_i^2 * (2 h_i + 3 h_{i-1}) / 12] * g''(xi_i)

with h_i = sigma_i - sigma_{i+1}. The total certified discretization residual
is sequence-level:

    D_s^AB(Pi_N) = Sum_{i=q-1}^{N-1} A_i^2 * | K_{s,i}^q(Pi_N) |^2 * | g^{(q)}(xi_i) |^2.

The optimal grid is

    Pi_{N,s}^* = argmin_{Pi_N} D_s^AB(Pi_N),

subject to monotone-grid constraints sigma_0 = sigma_max, sigma_N = sigma_min,
and sigma_i > sigma_{i+1}. This is a small numerical optimization (K-1 free
variables) that we solve with L-BFGS-B on a softmax-of-positive-deltas
parameterization to keep monotonicity automatic.

The pointwise rule m_s*(r) ∝ d_s(r)^{1/(p+1)} is recovered only when the
residual separates; for true AB-class solvers it does not, and Theorem B
in the appendix is the correct optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# AB-q kernel: per-interval kernel coefficient K_{i,q}(Pi) for a given grid
# ---------------------------------------------------------------------------

def ab2_kernel_coef(sigmas: np.ndarray) -> np.ndarray:
    """Per-interval AB-2 kernel coefficient on a descending sigma grid.

    sigmas: shape (N+1,), descending; sigmas[0] = sigma_max, sigmas[-1] = sigma_min.
    Returns array of shape (N-1,) -- the AB-2 update requires history so the
    first interval [sigmas[0], sigmas[1]] uses a 1st-order start (kernel = 0).
    """
    h = sigmas[:-1] - sigmas[1:]              # shape (N,), all positive
    h_curr = h[1:]                            # h_i for i = 1..N-1
    h_prev = h[:-1]                           # h_{i-1}
    # |e_i^AB2| = |h_i^2 (2 h_i + 3 h_{i-1}) / 12| * |g''|
    K = h_curr ** 2 * (2 * h_curr + 3 * h_prev) / 12.0
    return K


def ab4_kernel_coef(sigmas: np.ndarray) -> np.ndarray:
    """Per-interval AB-4 kernel coefficient via numerical integration.

    For q=4, K_{i,4} = (1/4!) * | int_{sigma_i}^{sigma_{i+1}}
        prod_{ell=0}^{3} (sigma - sigma_{i-ell}) d sigma |.

    Uses a 6-point Gauss-Legendre rule per interval (exact for polynomials
    up to degree 11; the integrand here is degree 4). Returns array of
    shape (N-3,) -- AB-4 needs 3 previous nodes plus the current, so the
    first 3 intervals are warmup and get kernel = 0.
    """
    n_intervals = sigmas.shape[0] - 1
    if n_intervals < 4:
        return np.zeros(0)
    # 6-point Gauss-Legendre on [-1, 1]
    gl_nodes = np.array([-0.9324695142, -0.6612093865, -0.2386191861,
                          0.2386191861, 0.6612093865, 0.9324695142])
    gl_weights = np.array([0.1713244924, 0.3607615730, 0.4679139346,
                           0.4679139346, 0.3607615730, 0.1713244924])
    K = np.zeros(n_intervals - 3)
    for i in range(3, n_intervals):
        a, b = sigmas[i], sigmas[i + 1]      # a > b for descending grid
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)                  # negative
        x_q = mid + half * gl_nodes           # sample points in [b, a]
        # prod_{ell=0..3} (x - sigmas[i-ell])
        prod = np.ones_like(x_q)
        for ell in range(4):
            prod = prod * (x_q - sigmas[i - ell])
        integral = (b - a) * 0.5 * np.sum(gl_weights * prod)
        K[i - 3] = abs(integral) / 24.0       # 1/4!
    return K


KERNEL_FNS: dict[int, Callable[[np.ndarray], np.ndarray]] = {
    2: ab2_kernel_coef,
    4: ab4_kernel_coef,
}


# ---------------------------------------------------------------------------
# g^{(q)} estimator on a fine sigma grid
# ---------------------------------------------------------------------------

@dataclass
class GqEstimate:
    sigma_grid: np.ndarray           # fine descending sigma grid
    gq_sq_per_sigma: np.ndarray      # ||g^{(q)}(sigma)||^2 averaged over calib batch
    q: int                           # AB order
    meta: dict
    # Lower-order derivatives for warmup step error (g' for Euler warmup, etc.)
    # Each entry: order p in 1..q-1 -> ||g^(p)(sigma)||^2 on the same sigma_grid.
    lower_order: dict[int, np.ndarray] | None = None


@torch.inference_mode()
def estimate_gq_squared(
    net,
    *,
    q: int = 2,
    num_calib_samples: int = 16,
    num_fine_sigmas: int = 96,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    seed: int = 0xAB,
    device="cuda",
    image_shape=None,
) -> GqEstimate:
    """Estimate ||g^{(q)}(sigma)||^2 at each of `num_fine_sigmas` sigma points.

    Strategy: g(sigma, x) = (x - denoised(x, sigma)) / sigma. We sample a
    Karras trajectory of `num_calib_samples` images, hold x fixed at each
    target sigma (using the Karras-trajectory nearest x), then evaluate g
    on a tight (sigma - 2 delta, sigma - delta, sigma, sigma + delta, sigma + 2 delta)
    stencil for finite differences. delta is chosen as 5%% of sigma to be in
    the smooth regime.

    For q=2 we use a centered 3-point stencil: g'' ~ (g(sigma+delta) - 2 g(sigma) + g(sigma-delta)) / delta^2.
    For q=4 we use a centered 5-point stencil:  g^(4) ~ (g_{-2} - 4 g_{-1} + 6 g_0 - 4 g_{+1} + g_{+2}) / delta^4.

    Returns an array of `||g^{(q)}||^2` (averaged over the calibration
    batch and over pixel dimensions) at each of the `num_fine_sigmas`
    sigma points.
    """
    from ._common import (
        denoise,
        karras_sigmas,
        resolve_shape,
        sample_initial_noise,
    )

    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    # Build a tight (no zero) sigma grid for the estimator.
    log_sig = np.linspace(np.log(sigma_max), np.log(sigma_min), num_fine_sigmas)
    sigma_grid = np.exp(log_sig).astype(np.float32)

    # Karras trajectory for in-distribution x at each sigma.
    karras_grid = karras_sigmas(num_fine_sigmas - 1, sigma_min, sigma_max, device=device).to(torch.float32)
    x = sample_initial_noise((num_calib_samples, *shape), float(karras_grid[0]),
                             seed=seed, device=device)
    karras_xs = [x]
    for i in range(num_fine_sigmas - 1):
        sa, sb = karras_grid[i], karras_grid[i + 1]
        # 2nd-order Heun to advance
        denoised = denoise(net, x, sa)
        d = (x - denoised) / sa
        x_next = x + (sb - sa) * d
        if sb.item() > 0:
            denoised2 = denoise(net, x_next, sb)
            d2 = (x_next - denoised2) / sb
            x_next = x + (sb - sa) * 0.5 * (d + d2)
        x = x_next
        karras_xs.append(x)

    karras_np = karras_grid.cpu().numpy()

    def x_at(sigma_target):
        idx = int(np.argmin(np.abs(karras_np - float(sigma_target))))
        return karras_xs[idx]

    def g(x_cur, sigma):
        sigma_t = torch.full((x_cur.shape[0],), float(sigma), device=device, dtype=x_cur.dtype)
        denoised = denoise(net, x_cur, sigma_t)
        return (x_cur - denoised) / float(sigma)

    g1_sq = np.zeros(num_fine_sigmas, dtype=np.float64)   # for Euler warmup
    if q == 2:
        # 3-point stencil: g'' = (g_+1 - 2 g_0 + g_-1) / delta^2
        gq_sq = np.zeros(num_fine_sigmas, dtype=np.float64)
        for j, sigma in enumerate(sigma_grid):
            delta = max(0.05 * float(sigma), 1e-4)
            sigma_lo = max(float(sigma) - delta, sigma_min * 0.5)
            sigma_hi = float(sigma) + delta
            x_cur = x_at(sigma)
            g0 = g(x_cur, sigma)
            gm = g(x_cur, sigma_lo)
            gp = g(x_cur, sigma_hi)
            actual_delta = 0.5 * (sigma_hi - sigma_lo)
            gpp = (gp - 2 * g0 + gm) / max(actual_delta ** 2, 1e-12)
            gprime = (gp - gm) / max(2 * actual_delta, 1e-12)
            gq_sq[j] = max(gpp.pow(2).mean().item(), 1e-30)
            g1_sq[j] = max(gprime.pow(2).mean().item(), 1e-30)
    elif q == 4:
        # 5-point stencil: g^(4) = (g_+2 - 4 g_+1 + 6 g_0 - 4 g_-1 + g_-2) / delta^4
        gq_sq = np.zeros(num_fine_sigmas, dtype=np.float64)
        for j, sigma in enumerate(sigma_grid):
            delta = max(0.05 * float(sigma), 1e-4)
            sigma_m2 = max(float(sigma) - 2 * delta, sigma_min * 0.5)
            sigma_m1 = max(float(sigma) - 1 * delta, sigma_min * 0.5)
            sigma_p1 = float(sigma) + delta
            sigma_p2 = float(sigma) + 2 * delta
            x_cur = x_at(sigma)
            g_m2 = g(x_cur, sigma_m2)
            g_m1 = g(x_cur, sigma_m1)
            g_0  = g(x_cur, sigma)
            g_p1 = g(x_cur, sigma_p1)
            g_p2 = g(x_cur, sigma_p2)
            g4 = (g_p2 - 4 * g_p1 + 6 * g_0 - 4 * g_m1 + g_m2) / max(delta ** 4, 1e-12)
            gprime = (g_p1 - g_m1) / max(2 * delta, 1e-12)
            gq_sq[j] = max(g4.pow(2).mean().item(), 1e-30)
            g1_sq[j] = max(gprime.pow(2).mean().item(), 1e-30)
    else:
        raise ValueError(f"only q in (2, 4) supported (got {q})")

    return GqEstimate(
        sigma_grid=sigma_grid,
        gq_sq_per_sigma=gq_sq,
        q=q,
        meta={
            "num_calib_samples": num_calib_samples,
            "num_fine_sigmas": num_fine_sigmas,
            "sigma_min": sigma_min, "sigma_max": sigma_max,
            "seed": seed,
        },
        lower_order={1: g1_sq},
    )


# ---------------------------------------------------------------------------
# Sequence optimization of the grid Pi_N
# ---------------------------------------------------------------------------

def _interp_gq_at(gq: GqEstimate, sigmas: np.ndarray, *, clip_pct: float = 95.0) -> np.ndarray:
    """log-sigma-interpolated lookup of ||g^{(q)}(sigma)||^2.

    The raw stencil estimate develops a huge spike near sigma=0 because
    g(x, sigma) = (x - denoised(x, sigma))/sigma has 1/sigma amplifying
    finite-difference noise. We clip g^{(q)}^2 to its `clip_pct`-th
    percentile to keep the sequence-level optimization well-conditioned.
    """
    log_target = np.log(np.clip(sigmas, gq.sigma_grid.min(), gq.sigma_grid.max()))
    log_src = np.log(gq.sigma_grid)
    values = gq.gq_sq_per_sigma
    if clip_pct < 100.0:
        cap = np.percentile(values, clip_pct)
        values = np.minimum(values, cap)
    return np.interp(log_target, log_src[::-1], values[::-1])


def _interp_g1_at(gq: GqEstimate, sigmas: np.ndarray) -> np.ndarray:
    """log-sigma-interpolated lookup of ||g'(sigma)||^2 (Euler warmup error)."""
    if gq.lower_order is None or 1 not in gq.lower_order:
        # fallback: assume g'^2 ~ g^(q)^2 (rough proxy)
        return _interp_gq_at(gq, sigmas)
    log_target = np.log(np.clip(sigmas, gq.sigma_grid.min(), gq.sigma_grid.max()))
    log_src = np.log(gq.sigma_grid)
    return np.interp(log_target, log_src[::-1], gq.lower_order[1][::-1])


def ab_residual_density(sigmas: np.ndarray, gq: GqEstimate, *,
                        sigma_floor: float | None = None,
                        amp_k: float = 2.0) -> float:
    """Sequence-level certified error for a q-step AB solver on grid `sigmas`.

    D_s^AB(Pi_N) = Sum_i A_i^2 * |K_i|^2 * ||g^{(p)}(xi_i)||^2

    where:
      - For i < q-1 (warmup): K_i = h_i^2/2, p = 1 (Euler local error).
      - For i >= q-1 (AB-q): K_i is the q-step interpolation-kernel coef, p = q.
      - A_i = xi_i^{-amp_k} is the terminal amplification (Karras-style
        perceptual weight; same role as the appendix's A_i, motivated by
        low-sigma FID sensitivity). amp_k = 2 by default (matches Round 2b
        validation tuning).

    The last interval ending at sigma=0 is excluded.
    """
    if sigma_floor is None:
        sigma_floor = float(gq.sigma_grid.min())
    q = gq.q
    n_intervals = sigmas.shape[0] - 1
    if n_intervals < 1:
        return 0.0
    total = 0.0
    h = sigmas[:-1] - sigmas[1:]
    sigma_lefts_full = sigmas[:-1]
    sigma_rights_full = sigmas[1:]
    for i in range(min(q - 1, n_intervals)):
        sl, sr = float(sigma_lefts_full[i]), float(sigma_rights_full[i])
        if sr <= sigma_floor * 0.9999:
            continue
        xi = np.exp(0.5 * (np.log(max(sl, sigma_floor))
                           + np.log(max(sr, sigma_floor))))
        g1_at = _interp_g1_at(gq, np.array([xi]))[0]
        A = xi ** (-amp_k)
        K_eu = (float(h[i]) ** 2) / 2.0
        total += (A ** 2) * (K_eu ** 2) * g1_at
    K_arr = KERNEL_FNS[q](sigmas)
    if K_arr.shape[0] > 0:
        sigma_lefts = sigmas[q - 1 : -1]
        sigma_rights = sigmas[q:]
        keep = sigma_rights > sigma_floor * 0.9999
        if np.any(keep):
            lefts = sigma_lefts[keep]
            rights = sigma_rights[keep]
            Kk = K_arr[keep]
            xis = np.exp(0.5 * (np.log(np.clip(lefts, sigma_floor, None))
                                + np.log(np.clip(rights, sigma_floor, None))))
            gq_at = _interp_gq_at(gq, xis)
            A = xis ** (-amp_k)
            total += float(np.sum(A ** 2 * Kk ** 2 * gq_at))
    return float(total)


def optimal_ab_grid(gq: GqEstimate, K: int, *, sigma_min=None, sigma_max=None,
                    init: str = "karras", maxiter: int = 200,
                    amp_k: float = 2.0) -> np.ndarray:
    """Find Pi_K = {sigma_0 = sigma_max, ..., sigma_K = 0} that minimizes
    D_s^AB(Pi_K) by L-BFGS-B.

    Parameterization: free deltas in R^K, softmax to get K positive gaps
    in log-sigma between sigma_max and sigma_min that sum to log_range.
    Grid is then (sigma_max, K-1 interior, sigma_min, 0) -- length K+2 so
    the AB solver can take K steps that integrate from sigma_max down to
    sigma_min and a final 1st-order fallback step to sigma=0.
    """
    from scipy import optimize as _opt
    sigma_min = float(sigma_min if sigma_min is not None else gq.sigma_grid.min())
    sigma_max = float(sigma_max if sigma_max is not None else gq.sigma_grid.max())
    log_range = np.log(sigma_max) - np.log(sigma_min)

    def params_to_sigmas(deltas: np.ndarray) -> np.ndarray:
        # softmax(deltas) gives K positive gaps in log-sigma summing to 1
        d = deltas - np.max(deltas)
        ex = np.exp(d)
        gaps = ex / ex.sum()
        log_sigmas_inner = np.log(sigma_max) - np.cumsum(gaps) * log_range   # length K
        sigmas = np.empty(K + 2, dtype=np.float64)
        sigmas[0] = sigma_max
        sigmas[1:K + 1] = np.exp(log_sigmas_inner)
        sigmas[K + 1] = 0.0
        return sigmas

    def loss(deltas: np.ndarray) -> float:
        sigmas = params_to_sigmas(deltas)
        return ab_residual_density(sigmas, gq, sigma_floor=sigma_min, amp_k=amp_k)

    if init == "karras":
        from ._common import karras_sigmas
        with torch.no_grad():
            kar = karras_sigmas(K, sigma_min, sigma_max, device="cpu").cpu().numpy()
        # karras_sigmas returns K+1 values from sigma_max..sigma_min then 0
        # drop trailing 0 to get K+1 monotone-positive sigmas (sigma_max..sigma_min)
        kar_inner = kar[:K + 1] if kar[-1] == 0 else kar
        if kar_inner.shape[0] != K + 1:
            # fall back to uniform-log init
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
    sigmas_opt = params_to_sigmas(res.x).astype(np.float32)
    return sigmas_opt
