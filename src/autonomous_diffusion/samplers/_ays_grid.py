"""AYS (Align Your Steps) schedule optimization -- faithful reimplementation.

Reference: Sabour, Fidler, Kreis. "Align Your Steps: Optimizing Sampling
Schedules in Diffusion Models", ICML 2024.  arXiv:2404.14507.

This implementation follows Sec. 3 of the paper directly:

  (1) Gaussian closed-form KLUB (Lemma 3.3, Eq. 14).  Assume p_data =
      N(0, c**2 I) and use the analytic ideal-denoiser integrand --
      not a Monte-Carlo estimate from the trained network.  With
      sigma(t) = t and s(t) = 1 (EDM convention), the per-segment KLUB
      over [t_{i-1}, t_i] is

          KLUB_i = integral_{t_{i-1}}^{t_i} (1/t^3) *
                       (1/(t^2 + c^2) - 1/(t_i^2 + c^2)) dt

      which is closed-form in t via partial fractions (see _klub_segment
      below).

  (2) Hierarchical subdivision (Sec. 3.3, last paragraph).  Start with
      a K0 = 10-step schedule initialized from a heuristic, optimize all
      9 interior points.  Then double to 20 by inserting log-midpoints,
      freeze the original 11 points, and optimize only the 10 new
      midpoints.  Double again to 40 and optimize only the 20 newest
      midpoints.

  (3) Piecewise log-linear interpolation to arbitrary target K.  The
      paper's released schedules are produced by viewing the 40-step
      schedule as a function K0 = 40 -> sigma_k, sampling K+1 points
      from a log-linear interpolation between (k/K0, log(sigma_k)) pairs.

The c parameter is fixed at c = 0.5 per Fig. 3 of the paper -- the
authors use this value across datasets after observing that c = 1
(unit-variance data) over-collapses the optimal schedule.

We work internally in *ascending* t (paper convention: t_0 = t_min <
t_1 < ... < t_K = t_max).  The public entry point
`ays_descending_sigmas` flips and appends 0 to match our Karras-style
convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian closed-form KLUB integrand (Lemma 3.3, Eq. 14)
# ---------------------------------------------------------------------------

def _I1_antideriv(u: np.ndarray, c2: float) -> np.ndarray:
    """Antiderivative of 1 / (u^3 (u^2 + c^2)) -- partial fractions.

    1/(u^3 (u^2+c^2)) = -1/(c^4 u) + 1/(c^2 u^3) + u/(c^4 (u^2+c^2))
    => integral = (1/(2 c^4)) ln(1 + c^2/u^2) - 1/(2 c^2 u^2)
    """
    return 0.5 / (c2 * c2) * np.log1p(c2 / (u * u)) - 0.5 / (c2 * u * u)


def _klub_segment(t_lo: float, t_hi: float, c2: float) -> float:
    """Closed-form integral over [t_lo, t_hi] (t_lo < t_hi) of
        (1/u^3) * (1/(u^2 + c^2) - 1/(t_hi^2 + c^2))   du
    """
    # integral of 1/(u^3 (u^2+c^2)) du
    a = _I1_antideriv(np.asarray(t_hi, dtype=np.float64), c2) \
        - _I1_antideriv(np.asarray(t_lo, dtype=np.float64), c2)
    # integral of (1/(t_hi^2+c^2)) * (1/u^3) du = (1/(t_hi^2+c^2)) * (-1/(2 u^2))
    b_anti_hi = -0.5 / (t_hi * t_hi)
    b_anti_lo = -0.5 / (t_lo * t_lo)
    b = (b_anti_hi - b_anti_lo) / (t_hi * t_hi + c2)
    return float(a - b)


def klub_total(t_grid: np.ndarray, c2: float) -> float:
    """KLUB of an ascending schedule t_0 < t_1 < ... < t_K.

    Sum of per-segment closed-form integrals.  No network calls.
    """
    total = 0.0
    for i in range(t_grid.shape[0] - 1):
        total += _klub_segment(float(t_grid[i]), float(t_grid[i + 1]), c2)
    return total


# ---------------------------------------------------------------------------
# Hierarchical subdivision optimizer (Sec. 3.3)
# ---------------------------------------------------------------------------

def _optimize_full_grid(t_min: float, t_max: float, K: int, c2: float,
                        init: np.ndarray | None = None, maxiter: int = 400) -> np.ndarray:
    """Optimize the K-1 free interior points of an ascending K-step schedule.

    Parameterization: log-gaps as softmax over K free deltas summing to
    log(t_max / t_min).  Always monotone by construction.
    """
    from scipy import optimize as _opt
    log_range = np.log(t_max) - np.log(t_min)

    def deltas_to_grid(deltas: np.ndarray) -> np.ndarray:
        d = deltas - np.max(deltas)
        ex = np.exp(d)
        gaps = ex / ex.sum()
        log_pts = np.log(t_min) + np.cumsum(gaps) * log_range
        grid = np.empty(K + 1, dtype=np.float64)
        grid[0] = t_min
        grid[1:K + 1] = np.exp(log_pts)
        # last point should equal t_max up to fp noise -- force it
        grid[K] = t_max
        return grid

    def loss(deltas: np.ndarray) -> float:
        return klub_total(deltas_to_grid(deltas), c2)

    if init is None:
        # Karras rho=7 init (their default initializer)
        rho = 7.0
        idx = np.arange(K + 1)
        log_kar = (t_max ** (1 / rho)
                   + (idx / K) * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
        kar_asc = log_kar[::-1].astype(np.float64)
        gaps0 = np.diff(np.log(kar_asc))
        gaps0 = np.clip(gaps0, 1e-9, None)
        gaps0 = gaps0 / gaps0.sum()
        deltas0 = np.log(gaps0 + 1e-12)
        deltas0 -= deltas0.mean()
    else:
        deltas0 = init.copy()

    res = _opt.minimize(
        loss, deltas0, method="Nelder-Mead",
        options=dict(maxiter=maxiter * (K + 1), xatol=1e-5, fatol=1e-8,
                     adaptive=True, disp=False),
    )
    return deltas_to_grid(res.x)


def _subdivide_and_finetune(grid: np.ndarray, c2: float, maxiter: int = 200) -> np.ndarray:
    """Insert a log-midpoint between every consecutive pair, then optimize
    ONLY the inserted points (frozen-old, free-new).

    Parameterization of each new point: sigmoid in log-coordinate between
    its two frozen neighbors, so monotonicity is preserved.
    """
    from scipy import optimize as _opt
    log_grid = np.log(grid)
    N = grid.shape[0] - 1                       # number of segments before subdivide
    # Insert log-midpoints; new grid has 2N+1 points (= K' + 1 with K' = 2K)
    new_log = np.empty(2 * N + 1, dtype=np.float64)
    new_log[0::2] = log_grid                     # frozen points at even indices
    new_log[1::2] = 0.5 * (log_grid[:-1] + log_grid[1:])  # initial guess for new midpoints
    # Each new midpoint i (at index 2i+1) lies between frozen neighbors
    # log_grid[i] and log_grid[i+1].  Parameterize log_t_new = log_left +
    # sigmoid(delta) * (log_right - log_left).
    log_left = log_grid[:-1]
    log_right = log_grid[1:]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def deltas_to_grid(deltas: np.ndarray) -> np.ndarray:
        out = new_log.copy()
        out[1::2] = log_left + sigmoid(deltas) * (log_right - log_left)
        return np.exp(out)

    def loss(deltas: np.ndarray) -> float:
        return klub_total(deltas_to_grid(deltas), c2)

    # Init at midpoint (sigmoid(0) = 0.5)
    deltas0 = np.zeros(N, dtype=np.float64)
    res = _opt.minimize(
        loss, deltas0, method="Nelder-Mead",
        options=dict(maxiter=maxiter * (N + 1), xatol=1e-5, fatol=1e-8,
                     adaptive=True, disp=False),
    )
    return deltas_to_grid(res.x)


@dataclass
class AysSchedule:
    """Container for an AYS 40-step ascending schedule and resampling helpers."""
    t_grid: np.ndarray        # ascending, length 41, t_0 = t_min, t_40 = t_max
    c2: float                 # squared importance-sampling scale (= c^2)
    meta: dict

    def resample_to_K(self, K: int) -> np.ndarray:
        """Piecewise log-linear interpolation in index domain from the
        cached 40-step schedule to K+1 ascending points.
        """
        K0 = self.t_grid.shape[0] - 1
        if K == K0:
            return self.t_grid.copy()
        x_src = np.arange(K0 + 1, dtype=np.float64) / K0
        y_src = np.log(self.t_grid)
        x_tgt = np.arange(K + 1, dtype=np.float64) / K
        y_tgt = np.interp(x_tgt, x_src, y_src)
        return np.exp(y_tgt)


def optimize_ays_40step(t_min: float, t_max: float, c: float = 0.5,
                        maxiter_full: int = 400,
                        maxiter_subdivide: int = 200) -> AysSchedule:
    """Run the full 10 -> 20 -> 40 subdivision protocol (Sec. 3.3).

    Returns a 40-step ascending schedule with t_0 = t_min, t_40 = t_max.
    """
    c2 = float(c * c)
    grid_10 = _optimize_full_grid(t_min, t_max, K=10, c2=c2, maxiter=maxiter_full)
    grid_20 = _subdivide_and_finetune(grid_10, c2=c2, maxiter=maxiter_subdivide)
    grid_40 = _subdivide_and_finetune(grid_20, c2=c2, maxiter=maxiter_subdivide)
    klub_10 = klub_total(grid_10, c2)
    klub_20 = klub_total(grid_20, c2)
    klub_40 = klub_total(grid_40, c2)
    return AysSchedule(
        t_grid=grid_40,
        c2=c2,
        meta={
            "t_min": t_min,
            "t_max": t_max,
            "c": c,
            "klub_10": klub_10,
            "klub_20": klub_20,
            "klub_40": klub_40,
            "stages": [grid_10.tolist(), grid_20.tolist(), grid_40.tolist()],
        },
    )


# ---------------------------------------------------------------------------
# Public entry point used by samplers: descending sigma grid (Karras format)
# ---------------------------------------------------------------------------

def ays_descending_sigmas(schedule: AysSchedule, K: int) -> np.ndarray:
    """Return a Karras-format sigma grid with `K` substantive descending
    sigmas (covering [sigma_min, sigma_max]) and a trailing 0, for a total
    of K+1 elements.  This matches `karras_sigmas(num_steps=K)`'s shape so
    our UniPC consumes exactly K NFE per sample.

    The substantive sigmas are sampled from the cached 40-step AYS
    schedule by piecewise log-linear interpolation in index space (the
    paper's resampling rule for sub-40-step schedules, Sec. 3.3 last
    paragraph).
    """
    K0 = schedule.t_grid.shape[0] - 1
    # K substantive points covering both endpoints
    x_src = np.arange(K0 + 1, dtype=np.float64) / K0
    y_src = np.log(schedule.t_grid)
    x_tgt = np.linspace(0.0, 1.0, K, dtype=np.float64)
    asc = np.exp(np.interp(x_tgt, x_src, y_src))
    desc = asc[::-1]
    return np.concatenate([desc.astype(np.float32), np.array([0.0], dtype=np.float32)])
