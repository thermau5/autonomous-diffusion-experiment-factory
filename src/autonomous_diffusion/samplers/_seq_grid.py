"""Sequence-coupled optimal grid for general-q multistep / predictor-corrector
/ EMS-modulated solvers.

Implements PDF 1 v3 Appendix B (revised): the unified Theorem B. Supersedes
the q=2-only `_ab_grid.py`. Three public pieces matching PDF v3 Sec. B.9:

  - kernel_coef(q, hs, W=None, corrector=False, gamma_p=1, gamma_c=0)
        closed-form (or EMS-quadrature) interpolation-kernel coefficient
        C_q^P (eq 17/18), C_q^C (eq 22/23), EMS-weighted (eq 29/30).

  - estimate_seq_table(net, ...)
        V_{q,i} = Var[g^{(q)}] (eq 39) on a fine reference trajectory via
        the general-q finite-difference stencil (eq 60/61), plus the
        terminal amplification A_i (eq 36/37) from a finite-difference JVP
        estimate of the integrand Jacobian log-norm L_g.

  - optimal_seq_grid(table, K, q_map, ...)
        sequence optimizer (eq 43) with cumulative-softmax gaps and a
        per-step order map q_i (eq 33/34), handling lower-order-final and
        pseudo-order via q_map.

Coordinate convention: PDF v3 §B.1 uses r in [0,1] with r=0 at sigma_max,
r=1 at sigma_min. We work directly in sigma here (descending grid
sigma_0=sigma_max ... sigma_N=sigma_min); the kernel coefficient depends
only on the gaps h_i = sigma_i - sigma_{i+1}, so it is coordinate-agnostic
given the gaps. (Working in sigma matches the solver implementations.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Interpolation-kernel coefficients (PDF v3 eq 17/18 predictor, 22/23 corrector)
# ---------------------------------------------------------------------------

def _predictor_integrand_poly(q: int, hs: np.ndarray) -> np.poly1d:
    """Polynomial in local coordinate x = r - r_i for the q-step AB predictor
    residual kernel:  x * prod_{j=1}^{q-1} (x + H_j),  H_j = sum_{l=1}^j h_{i-l}.

    hs = [h_i, h_{i-1}, ..., h_{i-q+1}] (length q).
    """
    # H_j cumulative sums of the *history* gaps hs[1], hs[2], ...
    poly = np.poly1d([1.0, 0.0])              # x
    H = 0.0
    for j in range(1, q):
        H += hs[j]
        poly = poly * np.poly1d([1.0, H])     # (x + H_j)
    return poly


def _corrector_integrand_poly(q: int, hs: np.ndarray) -> np.poly1d:
    """Polynomial for the corrector residual kernel (PDF v3 eq 23):
        (x - h_i) * x * prod_{j=1}^{q-2} (x + H_j).
    hs = [h_i, h_{i-1}, ..., h_{i-q+2}] (length q-1 used).
    """
    h_i = hs[0]
    poly = np.poly1d([1.0, -h_i]) * np.poly1d([1.0, 0.0])   # (x - h_i) * x
    H = 0.0
    for j in range(1, q - 1):
        H += hs[j]
        poly = poly * np.poly1d([1.0, H])
    return poly


def _integrate_poly_0_to_h(poly: np.poly1d, h_i: float) -> float:
    """Definite integral of `poly` over x in [0, h_i]."""
    anti = np.polyint(poly)
    return float(anti(h_i) - anti(0.0))


def _integrate_weighted(poly: np.poly1d, h_i: float, W: Callable[[float], float],
                        n_quad: int = 12) -> float:
    """Gauss-Legendre integral of W(x) * poly(x) over [0, h_i] for EMS weights."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # map [-1,1] -> [0, h_i]
    x = 0.5 * h_i * (nodes + 1.0)
    jac = 0.5 * h_i
    vals = np.array([W(float(xx)) for xx in x]) * poly(x)
    return float(jac * np.sum(weights * vals))


def kernel_coef(q: int, hs: np.ndarray, *, W: Callable[[float], float] | None = None,
                corrector: bool = False, gamma_p: float = 1.0,
                gamma_c: float = 0.0) -> float:
    """Combined interpolation-kernel coefficient at one step (PDF v3 eq 31 core).

    C = gamma_p * C_q^{P} + gamma_c * C_q^{C}, each = (1/q!) integral of the
    residual kernel polynomial (EMS-weighted by W if given).

    hs : [h_i, h_{i-1}, ..., h_{i-q+1}] (predictor needs q gaps; corrector
         needs q-1). Pass the full history; we slice as needed.
    W  : EMS weight function of local x (already folded l,s,b); None => no EMS.
    """
    h_i = float(hs[0])
    if q < 1:
        raise ValueError("q must be >= 1")
    fact = float(np.math.factorial(q)) if hasattr(np.math, "factorial") else float(np.prod(range(1, q + 1)) or 1)
    # predictor
    if q == 1:
        # C_1^P = integral_0^{h} x dx = h^2/2  (PDF v3 eq 35)
        cp = (h_i ** 2) / 2.0
    else:
        poly_p = _predictor_integrand_poly(q, hs)
        cp = (_integrate_weighted(poly_p, h_i, W) if W is not None
              else _integrate_poly_0_to_h(poly_p, h_i)) / fact
    total = gamma_p * cp
    # corrector (needs q >= 2)
    if gamma_c != 0.0 and q >= 2:
        poly_c = _corrector_integrand_poly(q, hs)
        cc = (_integrate_weighted(poly_c, h_i, W) if W is not None
              else _integrate_poly_0_to_h(poly_c, h_i)) / fact
        total += gamma_c * cc
    return total


# ---------------------------------------------------------------------------
# General-q finite-difference stencil (PDF v3 eq 60/61)
# ---------------------------------------------------------------------------

def finite_difference_stencil(q: int, L: int | None = None) -> tuple[np.ndarray, int]:
    """Central stencil coefficients c_{q,l} for the q-th derivative on a
    uniform grid (radius L >= ceil(q/2)). Solves the linear system

        sum_l c_l * l^a = 0   for a < q,
        sum_l c_l * l^q = q!.

    Returns (coeffs over l=-L..L, L). Derivative estimate is
        g^{(q)}(x_j) ~ delta^{-q} * sum_l c_l g_{j+l}.
    """
    if L is None:
        L = max(int(np.ceil(q / 2)), 1)
        # need at least q+1 nodes for a q-th derivative
        while (2 * L + 1) < (q + 1):
            L += 1
    ls = np.arange(-L, L + 1)
    n = ls.shape[0]
    # Vandermonde A[a, k] = ls[k]^a, a = 0..n-1
    A = np.vstack([ls.astype(np.float64) ** a for a in range(n)])
    b = np.zeros(n)
    b[q] = float(np.math.factorial(q)) if hasattr(np.math, "factorial") else float(np.prod(range(1, q + 1)) or 1)
    coeffs = np.linalg.solve(A, b)
    return coeffs, L


# ---------------------------------------------------------------------------
# Sequence table: V_{q,i} and amplification A_i on a fine reference grid
# ---------------------------------------------------------------------------

@dataclass
class SeqTable:
    sigma_grid: np.ndarray                       # fine descending sigma grid (length M)
    Vq: dict[int, np.ndarray]                    # q -> Var[g^{(q)}](sigma), length M
    logA: np.ndarray                             # cumulative log-amplification at each sigma
    meta: dict = field(default_factory=dict)
    ems: Callable[[float, float], float] | None = None  # (lambda, x_local) -> W weight

    def Vq_at(self, q: int, sigmas: np.ndarray, clip_pct: float = 97.0) -> np.ndarray:
        vals = self.Vq[q]
        if clip_pct < 100.0:
            vals = np.minimum(vals, np.percentile(vals, clip_pct))
        log_src = np.log(self.sigma_grid)[::-1]
        log_t = np.log(np.clip(sigmas, self.sigma_grid.min(), self.sigma_grid.max()))
        return np.interp(log_t, log_src, vals[::-1])

    def logA_at(self, sigmas: np.ndarray) -> np.ndarray:
        log_src = np.log(self.sigma_grid)[::-1]
        log_t = np.log(np.clip(sigmas, self.sigma_grid.min(), self.sigma_grid.max()))
        return np.interp(log_t, log_src, self.logA[::-1])


@torch.inference_mode()
def estimate_seq_table(
    net,
    *,
    q_values: tuple[int, ...] = (2, 3),
    num_calib_samples: int = 16,
    num_fine_sigmas: int = 128,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    n_jvp_probes: int = 2,
    seed: int = 0x5E9,
    device="cuda",
    image_shape=None,
) -> SeqTable:
    """Estimate V_{q,i}=Var[g^{(q)}] for each q in q_values plus the terminal
    amplification logA_i along a Heun-Karras reference trajectory.

    Amplification (PDF v3 eq 36/37): A_i = prod_{j>i} J_j, bounded by
    exp(integral L_g dτ). We estimate the local growth rate L_g(sigma) from a
    finite-difference JVP of the integrand g(x,sigma)=(x-D(x,sigma))/sigma:
        L_g(sigma) ~ mean_v log|J_g v| / |v|,  J_g = d g / d x,
    then accumulate logA(sigma) = integral_{sigma}^{sigma_min-side} L_g.
    """
    from ._common import denoise, karras_sigmas, resolve_shape, sample_initial_noise

    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    log_sig = np.linspace(np.log(sigma_max), np.log(sigma_min), num_fine_sigmas)
    sigma_grid = np.exp(log_sig).astype(np.float64)

    # Heun-Karras reference trajectory for in-distribution x at each sigma.
    kar = karras_sigmas(num_fine_sigmas - 1, sigma_min, sigma_max, device=device).to(torch.float32)
    x = sample_initial_noise((num_calib_samples, *shape), float(kar[0]), seed=seed, device=device)
    xs = [x]
    for i in range(num_fine_sigmas - 1):
        sa, sb = kar[i], kar[i + 1]
        den = denoise(net, x, sa)
        d = (x - den) / sa
        xn = x + (sb - sa) * d
        if sb.item() > 0:
            den2 = denoise(net, xn, sb)
            d2 = (xn - den2) / sb
            xn = x + (sb - sa) * 0.5 * (d + d2)
        x = xn
        xs.append(x)
    kar_np = kar.cpu().numpy()

    def x_at(sigma):
        return xs[int(np.argmin(np.abs(kar_np - float(sigma))))]

    def g_of(x_cur, sigma):
        st = torch.full((x_cur.shape[0],), float(sigma), device=device, dtype=x_cur.dtype)
        return (x_cur - denoise(net, x_cur, st)) / float(sigma)

    # log-uniform stencil step in log-sigma; convert to local sigma delta per point.
    Vq: dict[int, np.ndarray] = {q: np.zeros(num_fine_sigmas) for q in q_values}
    Lg = np.zeros(num_fine_sigmas)
    gen = torch.Generator(device=device).manual_seed(seed + 7)

    max_q = max(q_values)
    for j, sigma in enumerate(sigma_grid):
        delta = max(0.04 * float(sigma), 1e-4)
        x_cur = x_at(sigma)
        # build a symmetric stencil of g samples around sigma for the largest q
        radius = max(int(np.ceil(max_q / 2)), 1)
        while (2 * radius + 1) < (max_q + 1):
            radius += 1
        offsets = np.arange(-radius, radius + 1)
        g_samples = []
        for off in offsets:
            s_off = float(sigma) + off * delta
            s_off = max(s_off, sigma_min * 0.5)
            g_samples.append(g_of(x_cur, s_off))
        for q in q_values:
            coeffs, Lq = finite_difference_stencil(q, L=radius)
            # coeffs aligns with offsets (-radius..radius)
            gq = sum(float(c) * g_samples[k] for k, c in enumerate(coeffs)) / (delta ** q)
            # variance across batch+pixels (centered): Var = E[||.||^2] - ||E[.]||^2
            flat = gq.reshape(gq.shape[0], -1)
            mean_vec = flat.mean(dim=0)
            var = (flat - mean_vec).pow(2).mean().item()
            Vq[q][j] = max(var, 1e-30)
        # amplification log-norm of the FLOW Jacobian J_f = d f/d x, where
        # f = (x - D(x,sigma))/sigma is the EDM probability-flow RHS. Use the
        # Rayleigh quotient mu = <v, J_f v>/<v,v> (the logarithmic norm /
        # spectral abscissa, which may be negative = local contraction), not
        # the operator norm |J_f v|/|v| (always >= 0). The terminal
        # amplification then integrates mu over d sigma with reverse-time sign
        # (PDF v3 eq 36/37 specialized to the EDM flow).
        mus = []
        for _ in range(n_jvp_probes):
            v = torch.randn(x_cur.shape, generator=gen, device=device, dtype=x_cur.dtype)
            vnorm = v.reshape(v.shape[0], -1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            v = v / vnorm
            eps = 1e-3 * float(sigma)
            g0 = g_of(x_cur, sigma)
            g1 = g_of(x_cur + eps * v, sigma)
            jv = (g1 - g0) / eps                      # ~ J_f v
            vflat = v.reshape(v.shape[0], -1)
            jvflat = jv.reshape(jv.shape[0], -1)
            rayleigh = (vflat * jvflat).sum(dim=1) / (vflat * vflat).sum(dim=1).clamp(min=1e-12)
            mus.append(rayleigh.mean().item())
        Lg[j] = float(np.mean(mus))               # mu(sigma): log-norm of J_f

    # Terminal amplification: a perturbation at sigma_i propagates DOWN to
    # sigma_min along the reverse flow dx/dsigma = f. The variational equation
    # gives |delta(sigma_min)|/|delta(sigma_i)| = exp(-int_{sigma_min}^{sigma_i} mu(J_f) dsigma).
    # Hence logA(sigma_i) = -int_{sigma_min}^{sigma_i} mu dsigma (measure d sigma,
    # reverse-time sign). sigma_grid is descending, so integrate from the
    # sigma_min end upward.
    logA = np.zeros(num_fine_sigmas)
    dsig = np.abs(np.diff(sigma_grid))             # |sigma_j - sigma_{j+1}|
    acc = 0.0
    for j in range(num_fine_sigmas - 2, -1, -1):
        acc += -0.5 * (Lg[j] + Lg[j + 1]) * dsig[j]
        logA[j] = acc

    return SeqTable(
        sigma_grid=sigma_grid,
        Vq=Vq,
        logA=logA,
        meta={"num_calib_samples": num_calib_samples, "num_fine_sigmas": num_fine_sigmas,
              "sigma_min": sigma_min, "sigma_max": sigma_max, "q_values": list(q_values),
              "n_jvp_probes": n_jvp_probes, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Sequence optimizer (PDF v3 eq 43)
# ---------------------------------------------------------------------------

def seq_residual(sigmas: np.ndarray, table: SeqTable, q_map: Callable[[int, int], int],
                 *, ems_W: Callable[[float, np.ndarray], Callable[[float], float]] | None = None,
                 gamma_p: float = 1.0, gamma_c: float = 0.0,
                 sigma_floor: float | None = None) -> float:
    """D_s^class(Pi_N) = sum_i A_i^2 |C_{q_i,i}|^2 V_{q_i,i}  (PDF v3 eq 41)."""
    if sigma_floor is None:
        sigma_floor = float(table.sigma_grid.min())
    N = sigmas.shape[0] - 1
    h = sigmas[:-1] - sigmas[1:]
    total = 0.0
    for i in range(N):
        sr = float(sigmas[i + 1])
        if sr <= sigma_floor * 0.9999:
            continue
        q_i = int(q_map(i, N))
        # history gaps [h_i, h_{i-1}, ..., h_{i-q+1}], clamped at trajectory start
        hist = []
        for j in range(q_i):
            idx = i - j
            hist.append(float(h[idx]) if idx >= 0 else float(h[0]))
        hs = np.array(hist, dtype=np.float64)
        xi = np.exp(0.5 * (np.log(max(float(sigmas[i]), sigma_floor))
                           + np.log(max(sr, sigma_floor))))
        W = None
        if ems_W is not None:
            W = ems_W(xi, hs)
        C = kernel_coef(q_i, hs, W=W, corrector=(gamma_c != 0.0),
                        gamma_p=gamma_p, gamma_c=gamma_c)
        V = float(table.Vq_at(q_i, np.array([xi]))[0])
        A2 = float(np.exp(2.0 * table.logA_at(np.array([xi]))[0]))
        total += A2 * (C ** 2) * V
    return float(total)


def optimal_seq_grid(table: SeqTable, K: int, q_map: Callable[[int, int], int],
                     *, sigma_min: float | None = None, sigma_max: float | None = None,
                     ems_W=None, gamma_p: float = 1.0, gamma_c: float = 0.0,
                     init: str = "karras", maxiter: int = 300,
                     include_zero_tail: bool = True) -> np.ndarray:
    """Minimize D_s^class over monotone grids via cumulative-softmax gaps.

    Returns descending sigmas. If include_zero_tail, appends 0.0 (UniPC-style
    boundary step); else the last sigma is sigma_min (DPM-Solver-v3 style).
    """
    from scipy import optimize as _opt
    sigma_min = float(sigma_min if sigma_min is not None else table.sigma_grid.min())
    sigma_max = float(sigma_max if sigma_max is not None else table.sigma_grid.max())
    log_range = np.log(sigma_max) - np.log(sigma_min)
    n_sub = K if include_zero_tail else K  # substantive sigmas span [smax, smin]

    def deltas_to_sigmas(deltas: np.ndarray) -> np.ndarray:
        d = deltas - np.max(deltas)
        ex = np.exp(d)
        gaps = ex / ex.sum()
        log_inner = np.log(sigma_max) - np.cumsum(gaps) * log_range   # length n_sub
        if include_zero_tail:
            sig = np.empty(K + 2, dtype=np.float64)
            sig[0] = sigma_max
            sig[1:K + 1] = np.exp(log_inner[:K])
            sig[K] = sigma_min   # force exact endpoint
            sig[K + 1] = 0.0
        else:
            sig = np.empty(K + 1, dtype=np.float64)
            sig[0] = sigma_max
            sig[1:K + 1] = np.exp(log_inner[:K])
            sig[K] = sigma_min
        return sig

    def loss(deltas):
        return seq_residual(deltas_to_sigmas(deltas), table, q_map,
                            ems_W=ems_W, gamma_p=gamma_p, gamma_c=gamma_c,
                            sigma_floor=sigma_min)

    if init == "karras":
        from ._common import karras_sigmas
        kar = karras_sigmas(n_sub, sigma_min, sigma_max, device="cpu").cpu().numpy()
        kar_inner = kar[:n_sub + 1] if kar[-1] == 0 else kar
        if kar_inner.shape[0] != n_sub + 1:
            kar_inner = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_sub + 1))
        log_gaps = -np.diff(np.log(np.clip(kar_inner, 1e-9, None))) / log_range
        g0 = np.clip(log_gaps, 1e-9, None)
        g0 = g0 / g0.sum()
        deltas0 = np.log(g0 + 1e-9)
        deltas0 -= deltas0.mean()
    else:
        deltas0 = np.zeros(n_sub, dtype=np.float64)

    res = _opt.minimize(loss, deltas0, method="Nelder-Mead",
                        options=dict(maxiter=maxiter * (n_sub + 2), xatol=1e-5,
                                     fatol=1e-10, adaptive=True, disp=False))
    return deltas_to_sigmas(res.x).astype(np.float32)
