"""Aggregate ays_unipc metrics across (nfe, seed) into a summary.json that
matches the locked-test per_sampler structure, then print the 3-way
comparison table (UniPC-Karras vs UniPC-AYS vs proposed_unipc) for direct
inclusion in the LaTeX report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def find_ays_metrics(run_root: Path, dataset: str = "cifar10", phase: str = "test"):
    out = []
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if f"_{phase}_{dataset}_ays_unipc_" not in name:
            continue
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        out.append(json.loads(mp.read_text()))
    return out


def aggregate(metrics: list[dict]):
    by_nfe: dict[int, list[float]] = {}
    walls: dict[int, list[float]] = {}
    for m in metrics:
        n = int(m["nfe_per_sample"]) if m.get("nfe_per_sample") else int(m["nfe"])
        by_nfe.setdefault(n, []).append(float(m["clean_fid"]))
        if m.get("wall_seconds") is not None:
            walls.setdefault(n, []).append(float(m["wall_seconds"]))
    rows = []
    for nfe in sorted(by_nfe):
        fids = np.array(by_nfe[nfe], dtype=float)
        mean = float(fids.mean())
        sem = float(fids.std(ddof=1) / np.sqrt(fids.size)) if fids.size > 1 else 0.0
        wall = float(np.mean(walls[nfe])) if walls.get(nfe) else None
        rows.append({"nfe": nfe, "fid_mean": mean, "fid_sem": sem,
                     "per_seed_fid": fids.tolist(), "wall_seconds_mean": wall,
                     "n_seeds": int(fids.size)})
    return rows


def load_locked_unipc(locked_summary: Path):
    d = json.loads(locked_summary.read_text())
    ps = d["per_sampler"]
    out = {}
    for sid in ("unipc", "proposed_unipc"):
        out[sid] = {nfe: (fid, sem) for nfe, fid, sem in ps[sid]["frontier"]}
    return out


def main():
    repo = Path(__file__).resolve().parent.parent
    run_root = repo / "outputs" / "runs"
    locked = repo / "outputs" / "locked_test" / "20260524T154736_cifar10_locked_test" / "summary.json"
    out_dir = repo / "outputs" / "round5b_ays"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = find_ays_metrics(run_root)
    if not metrics:
        sys.exit("No ays_unipc metrics found yet.")
    rows = aggregate(metrics)
    locked_unipc = load_locked_unipc(locked)

    summary = {
        "sampler": "ays_unipc",
        "schedule": "ays_gaussian_closed_form (c=0.5, 10->20->40 subdivision, log-linear interp)",
        "solver_core": "unipc_2pc",
        "dataset": "cifar10",
        "samples_per_run": 10000,
        "n_seeds_target": 5,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")

    print()
    print("=== UniPC core (Karras) vs UniPC core (AYS, faithful reimpl) vs UniPC core (Ours, m_s*) ===")
    print(f"{'NFE':>4}  {'Karras':>17}  {'AYS-faithful':>17}  {'Ours (m_s*)':>17}  "
          f"{'AYS-Karras':>10}  {'Ours-Karras':>11}")
    for r in rows:
        nfe = r["nfe"]
        ays_mean, ays_sem = r["fid_mean"], r["fid_sem"]
        kar_mean, kar_sem = locked_unipc["unipc"].get(nfe, (float("nan"), float("nan")))
        our_mean, our_sem = locked_unipc["proposed_unipc"].get(nfe, (float("nan"), float("nan")))
        d_ays = ays_mean - kar_mean
        d_our = our_mean - kar_mean
        print(f"{nfe:>4}  {kar_mean:7.4f}+/-{kar_sem:5.4f}  "
              f"{ays_mean:7.4f}+/-{ays_sem:5.4f}  "
              f"{our_mean:7.4f}+/-{our_sem:5.4f}  "
              f"{d_ays:>+8.4f}  {d_our:>+9.4f}")


if __name__ == "__main__":
    main()
