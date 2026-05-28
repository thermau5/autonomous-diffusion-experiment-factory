"""Finite-N calibrated certificate (PDF 1 v4.1, Appendix C) -- on-the-fly grid.

The grid for a history-dependent solver is optimized by minimizing the
calibrated diagonal certificate

    D_s^cal(Pi_N) = sum_i A_i^2 * e-hat_s(sigma_i, h_i, h_{i-1}, ...)^2

where e-hat_s is the MEASURED one-step solver error against a dense
reference trajectory (Definition 1). Per v4.1 Proposition 3.1, the
estimator must not under-report the true surface; we therefore measure
e-hat on-the-fly at every candidate grid (no coarse interpolation), which
is the exact (not surrogate) certificate and has no exploitable blind
spots.

Coordinate: descending sigma. A K-NFE grid has K substantive sigmas
(sigma_max ... sigma_min) plus a trailing 0 boundary -> K+1 elements,
so the multistep solver consumes exactly K NFE.

This module is solver-agnostic in the one-step map `step_fn`; DEIS tAB-2
and DPM-Solver-v3 provide their own `step_fn`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ._common import denoise, karras_sigmas, resolve_shape, sample_initial_noise
from .proposed_control import _net_key


@dataclass
class SeqReference:
    sig: np.ndarray                 # dense descending sigma nodes, length M
    Xref: torch.Tensor              # [M, B, C, H, W] reference states
    EPSref: torch.Tensor            # [M, B, C, H, W] eps = (x - D)/sigma
    meta: dict

    def near(self, s: float) -> int:
        return int(np.argmin(np.abs(self.sig - s)))


_REF_CACHE: dict[str, SeqReference] = {}


@torch.inference_mode()
def build_reference(net, *, M: int = 1025, B: int = 64,
                    sigma_min: float = 0.002, sigma_max: float = 80.0,
                    seed: int = 0, device="cuda", image_shape=None) -> SeqReference:
    """Dense high-NFE Heun reference trajectory + eps at every node. Cached
    per-process by net key (the reference is the 'exact' flow used to
    measure one-step defects)."""
    key = _net_key(net)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    device = torch.device(device)
    shape = resolve_shape(net, image_shape)
    sig = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), M))
    sig_t = torch.tensor(sig, dtype=torch.float64, device=device)
    x = sample_initial_noise((B, *shape), float(sig_t[0]), seed=seed, device=device).to(torch.float64)
    xs = [x]
    for i in range(M - 1):
        sa, sb = sig_t[i], sig_t[i + 1]
        den = denoise(net, x, sa); d = (x - den) / sa; xn = x + (sb - sa) * d
        if sb.item() > 0:
            den2 = denoise(net, xn, sb); d2 = (xn - den2) / sb; xn = x + (sb - sa) * 0.5 * (d + d2)
        x = xn; xs.append(x)
    Xref = torch.stack(xs, 0)
    EPSref = torch.empty_like(Xref)
    for i in range(M):
        st = torch.full((B,), float(sig_t[i]), device=device, dtype=torch.float64)
        EPSref[i] = (Xref[i] - denoise(net, Xref[i], st)) / float(sig_t[i])
    ref = SeqReference(sig=sig, Xref=Xref, EPSref=EPSref,
                       meta={"M": M, "B": B, "sigma_min": sigma_min,
                             "sigma_max": sigma_max, "seed": seed})
    _REF_CACHE[key] = ref
    return ref


# ---------------------------------------------------------------------------
# DEIS tAB-2 one-step map for the calibrated objective
# ---------------------------------------------------------------------------

def _deis_onestep_from_ref(ref: SeqReference, ji: int, jn: int, jp: int | None):
    """One DEIS tAB-2 step from reference node ji to jn, prev node jp (or
    Euler if jp is None). Returns the stepped state (batch tensor)."""
    sig_t = torch.tensor(ref.sig, dtype=torch.float64, device=ref.Xref.device)
    si, sn = sig_t[ji], sig_t[jn]
    if jp is None:
        return ref.Xref[ji] + (sn - si) * ref.EPSref[ji]
    sp = sig_t[jp]
    li = -si.log(); ln = -sn.log(); lp = -sp.log()
    h_i = ln - li; h_p = li - lp
    cc = 1 + h_i / (2 * h_p); cp = -h_i / (2 * h_p)
    return ref.Xref[ji] + (sn - si) * (cc * ref.EPSref[ji] + cp * ref.EPSref[jp])


def deis_seq_objective(grid: np.ndarray, ref: SeqReference) -> float:
    """Calibrated diagonal certificate with A_i=1 (PDF v4.1 eq 6/31),
    measured on-the-fly. grid: K+1 elements [sigma_max,...,sigma_min,0]."""
    s = grid[:-1]                       # K substantive sigmas
    total = 0.0
    for i in range(len(s) - 1):
        ji = ref.near(s[i]); jn = ref.near(s[i + 1])
        jp = ref.near(s[i - 1]) if i > 0 else None
        xloc = _deis_onestep_from_ref(ref, ji, jn, jp)
        total += (xloc - ref.Xref[jn]).reshape(ref.Xref.shape[1], -1).pow(2).sum(1).mean().item()
    return total


def optimal_deis_seq_grid(ref: SeqReference, K: int, *, maxiter: int = 150) -> np.ndarray:
    """Minimize the calibrated certificate over K-substantive monotone grids.
    Returns K+1 elements [sigma_max, ..., sigma_min, 0] (NFE = K)."""
    from scipy import optimize as _opt
    sigma_min = float(ref.sig.min()); sigma_max = float(ref.sig.max())
    lr = np.log(sigma_max) - np.log(sigma_min)

    def d2s(deltas):
        d = deltas - deltas.max(); ex = np.exp(d); gaps = ex / ex.sum()
        loginner = np.log(sigma_max) - np.cumsum(gaps) * lr   # K-1 values; last = log(sigma_min)
        out = np.empty(K + 1)
        out[0] = sigma_max
        out[1:K] = np.exp(loginner[:K - 1])
        out[K - 1] = sigma_min
        out[K] = 0.0
        return out

    kar = karras_sigmas(K, sigma_min, sigma_max, device="cpu").cpu().numpy()
    ki = kar[:K] if kar[-1] == 0 else kar
    lg = -np.diff(np.log(np.clip(ki, 1e-9, None))) / lr
    g0 = np.clip(lg, 1e-9, None); g0 = g0 / g0.sum()
    d0 = np.log(g0 + 1e-9); d0 -= d0.mean()
    if d0.shape[0] != K - 1:
        d0 = np.zeros(K - 1)

    res = _opt.minimize(lambda d: deis_seq_objective(d2s(d), ref), d0, method="Nelder-Mead",
                        options=dict(maxiter=maxiter * K, xatol=1e-4, fatol=1e-8, adaptive=True))
    return d2s(res.x).astype(np.float32)


# ---------------------------------------------------------------------------
# Grid disk cache
# ---------------------------------------------------------------------------

def grid_cache_path(net, K: int, *, root: str | Path = "outputs/calibration",
                    tag: str = "deis_seq") -> Path:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    return root / f"{tag}_grid_{_net_key(net)}_K{K}.json"


def get_or_build_deis_seq_grid(net, K: int, *, root="outputs/calibration",
                               device="cuda", image_shape=None,
                               ref_kwargs: dict | None = None) -> np.ndarray:
    path = grid_cache_path(net, K, root=root)
    if path.exists():
        return np.array(json.loads(path.read_text())["grid"], dtype=np.float32)
    ref = build_reference(net, device=device, image_shape=image_shape, **(ref_kwargs or {}))
    grid = optimal_deis_seq_grid(ref, K)
    path.write_text(json.dumps({"grid": grid.tolist(), "K": K, "ref_meta": ref.meta}))
    return grid
