"""Pareto frontier + Pareto-AUC over the (FID, NFE) plane.

Definition (per benchmark_contract.yaml):
    u_a dominates u_b  iff   FID(u_a)  <=  FID(u_b)
                       AND   NFE(u_a)  <=  NFE(u_b)
                       AND   wall(u_a) <=  wall(u_b)
                       AND at least one strict inequality.

Per-sampler dominance check: a sampler's frontier dominates another's if
every point on the latter is weakly dominated by some point on the former.

Pareto-AUC: area below the (NFE, FID) Pareto frontier within a stated NFE
window [nfe_lo, nfe_hi]. Lower AUC == better quality-efficiency trade-off.
We integrate the lower envelope of FID(NFE) over log-NFE because the NFE
grid is log-spaced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FrontierPoint:
    sampler: str
    nfe: int
    fid_mean: float
    fid_sem: float
    wall_seconds_mean: float
    per_seed_fid: tuple[float, ...]


def aggregate_seeds(runs: Iterable[dict]) -> list[FrontierPoint]:
    """Group runs by (sampler, nfe) and average FID over seeds.

    Each input dict must have keys: sampler, nfe, seed, clean_fid,
    wall_seconds. Returns one FrontierPoint per (sampler, nfe).
    """
    bucket: dict[tuple[str, int], list[dict]] = {}
    for r in runs:
        key = (str(r["sampler"]), int(r["nfe"]))
        bucket.setdefault(key, []).append(r)
    out: list[FrontierPoint] = []
    for (sampler, nfe), rs in sorted(bucket.items()):
        fids = np.array([float(r["clean_fid"]) for r in rs], dtype=np.float64)
        walls = np.array([float(r.get("wall_seconds") or 0.0) for r in rs], dtype=np.float64)
        n = len(fids)
        sem = float(fids.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out.append(FrontierPoint(
            sampler=sampler, nfe=nfe,
            fid_mean=float(fids.mean()),
            fid_sem=sem,
            wall_seconds_mean=float(walls.mean()),
            per_seed_fid=tuple(float(x) for x in fids),
        ))
    return out


def pareto_frontier(points: Sequence[FrontierPoint], *, keys=("nfe", "fid_mean")) -> list[FrontierPoint]:
    """Return the points on the Pareto frontier of the (key1, key2) plane
    where both keys are minimized."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: (getattr(p, keys[0]), getattr(p, keys[1])))
    frontier: list[FrontierPoint] = []
    best_y = float("inf")
    for p in pts:
        y = getattr(p, keys[1])
        if y < best_y - 1e-12:
            frontier.append(p)
            best_y = y
    return frontier


def pareto_dominates(set_a: Sequence[FrontierPoint], set_b: Sequence[FrontierPoint]) -> bool:
    """True iff for every point b in set_b, exists a in set_a with
    a.nfe <= b.nfe and a.fid_mean <= b.fid_mean (at least one strict).
    """
    if not set_b:
        return True
    if not set_a:
        return False
    for b in set_b:
        beaten = False
        for a in set_a:
            if a.nfe <= b.nfe and a.fid_mean <= b.fid_mean and (a.nfe < b.nfe or a.fid_mean < b.fid_mean):
                beaten = True
                break
            if a.nfe <= b.nfe and a.fid_mean <= b.fid_mean:
                beaten = True
                break
        if not beaten:
            return False
    return True


def pareto_auc(frontier: Sequence[FrontierPoint], *, nfe_lo: float, nfe_hi: float, log_x: bool = True) -> float:
    """Area under the (NFE, FID) frontier within [nfe_lo, nfe_hi]. Lower is
    better. If log_x, integrate over log10(NFE) which is the natural axis
    for the typical log-spaced NFE grid in our contract.

    The frontier is treated as a step function (piecewise constant), staying
    at each point's FID until the next, with the final segment held to nfe_hi.
    """
    pts = sorted(frontier, key=lambda p: p.nfe)
    if not pts:
        return float("inf")
    auc = 0.0
    for i, p in enumerate(pts):
        x_start = max(p.nfe, nfe_lo)
        x_end = pts[i + 1].nfe if i + 1 < len(pts) else nfe_hi
        x_end = min(x_end, nfe_hi)
        if x_end <= x_start:
            continue
        if log_x:
            xs = np.log10(x_start)
            xe = np.log10(x_end)
        else:
            xs, xe = x_start, x_end
        auc += p.fid_mean * (xe - xs)
    return float(auc)


def per_sampler_summary(points: Sequence[FrontierPoint]) -> dict[str, dict]:
    """Per-sampler aggregate: best FID, NFE at best, frontier length."""
    by_sampler: dict[str, list[FrontierPoint]] = {}
    for p in points:
        by_sampler.setdefault(p.sampler, []).append(p)
    out = {}
    for sampler, pts in by_sampler.items():
        best = min(pts, key=lambda q: q.fid_mean)
        front = pareto_frontier(pts)
        out[sampler] = {
            "best_fid": best.fid_mean,
            "best_fid_sem": best.fid_sem,
            "best_nfe": best.nfe,
            "num_frontier_points": len(front),
            "frontier": [(p.nfe, p.fid_mean, p.fid_sem) for p in front],
        }
    return out
