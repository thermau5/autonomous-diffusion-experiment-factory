"""Pareto-frontier figure: FID vs NFE, log-x, per-sampler curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_COLORS = {
    "proposed_control": "tab:red",
    "edm_heun":         "tab:blue",
    "edm_euler":        "tab:cyan",
    "karras_schedule":  "tab:purple",
    "uniform_schedule": "tab:olive",
    "dpm_solver":       "tab:green",
    "dpm_solver_pp":    "tab:orange",
    "unipc":            "tab:brown",
    "deis":             "tab:pink",
    "pndm":             "tab:gray",
    "restart":          "lime",
    "ddim":             "navy",
    "ddpm_ancestral":   "darkgoldenrod",
}


def pareto_frontier_figure(
    summary: dict,
    *,
    out_path: str | Path,
    title: str = "CIFAR-10: FID vs NFE",
    ylim: tuple[float, float] | None = None,
) -> Path:
    per = summary.get("per_sampler", {})
    fig, ax = plt.subplots(figsize=(7, 5))
    for sampler, rec in sorted(per.items()):
        front = rec.get("frontier", [])
        if not front:
            continue
        xs = np.array([f[0] for f in front], dtype=float)
        ys = np.array([f[1] for f in front], dtype=float)
        sems = np.array([(f[2] if len(f) > 2 else 0.0) for f in front], dtype=float)
        color = _COLORS.get(sampler, None)
        is_proposed = sampler == "proposed_control"
        ax.errorbar(
            xs, ys, yerr=sems, marker="o" if not is_proposed else "D",
            label=sampler, linewidth=2.0 if is_proposed else 1.0,
            color=color, capsize=2.0, alpha=0.95 if is_proposed else 0.7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("NFE (network function evaluations per sample)")
    ax.set_ylabel("Clean-FID $\\downarrow$")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def write_figures_from_summary(summary_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(summary_path).read_text())
    paths = {}
    dset = summary.get("dataset", "?").upper()
    paths["pareto"] = pareto_frontier_figure(
        summary, out_path=out_dir / "pareto_fid_nfe.png",
        title=f"{dset}: Quality-efficiency frontier",
    )
    # Zoomed-in version: the interesting region is FID < 60. Cuts the
    # EDM-Heun NFE=5 (FID 343, 3 num_steps is too few) outlier so the
    # mid-NFE comparison is readable.
    paths["pareto_zoom"] = pareto_frontier_figure(
        summary, out_path=out_dir / "pareto_fid_nfe_zoom.png",
        title=f"{dset}: Quality-efficiency frontier (zoom, FID $\\leq$ 60)",
        ylim=(14, 60),
    )
    return paths
