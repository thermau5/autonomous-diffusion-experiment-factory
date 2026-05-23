"""LaTeX table generators from a sweep summary.json."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _fmt(x, sem=None, ndigits=2):
    if x is None or (isinstance(x, float) and not (x == x)):
        return "--"
    s = f"{x:.{ndigits}f}"
    if sem is not None and sem > 0:
        s += rf" {{\scriptsize $\pm$ {sem:.{ndigits}f}}}"
    return s


def main_results_table(summary: dict, *, samplers_order: list[str] | None = None) -> str:
    """LaTeX table: per-sampler best FID and best NFE; mean ± SEM."""
    per = summary.get("per_sampler", {})
    if samplers_order is None:
        samplers_order = sorted(per.keys(), key=lambda s: per[s]["best_fid"])
    rows = []
    for s in samplers_order:
        if s not in per:
            continue
        rec = per[s]
        rows.append(
            f"{_latex_sampler(s)} & {rec['best_nfe']} & "
            f"{_fmt(rec['best_fid'], rec.get('best_fid_sem'))} \\\\"
        )
    return (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Sampler & Best NFE & Clean-FID $\\downarrow$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )


def fid_at_matched_nfe_table(summary: dict, nfes: Iterable[int]) -> str:
    """LaTeX table: FID for each sampler at each NFE in `nfes`."""
    per = summary.get("per_sampler", {})
    nfes = list(nfes)
    samplers_order = sorted(per.keys())
    header = "Sampler & " + " & ".join(f"NFE={n}" for n in nfes) + " \\\\"
    rows = []
    for s in samplers_order:
        cells = [_latex_sampler(s)]
        frontier_dict = {fnfe: (fid, sem) for fnfe, fid, sem in per[s].get("frontier", [])}
        # frontier only stores Pareto points; fall back to scanning every run.
        all_pts = summary.get("_all_points", {}).get(s, {})
        for n in nfes:
            v = all_pts.get(str(n)) or all_pts.get(n)
            if v:
                cells.append(_fmt(v["fid_mean"], v.get("fid_sem")))
            elif n in frontier_dict:
                fid, sem = frontier_dict[n]
                cells.append(_fmt(fid, sem))
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")
    return (
        "\\begin{tabular}{l" + "c" * len(nfes) + "}\n\\toprule\n"
        + header + "\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    )


def pareto_auc_table(summary: dict) -> str:
    auc = summary.get("pareto_auc_fid_nfe_log")
    if auc is None:
        return ""
    return (
        "\\begin{tabular}{l c}\n\\toprule\n"
        "Quantity & Value \\\\\n\\midrule\n"
        f"Pareto-AUC (FID vs NFE, log-scaled) & {auc:.3f} \\\\\n"
        "\\bottomrule\n\\end{tabular}"
    )


def _latex_sampler(name: str) -> str:
    pretty = {
        "edm_euler":        "EDM-Euler",
        "edm_heun":         "EDM-Heun",
        "karras_schedule":  "Karras schedule",
        "uniform_schedule": "Uniform-log schedule",
        "ddim":             "DDIM",
        "ddpm_ancestral":   "DDPM ancestral",
        "dpm_solver":       "DPM-Solver",
        "dpm_solver_pp":    "DPM-Solver++",
        "unipc":            "UniPC",
        "deis":             "DEIS",
        "pndm":             "PNDM",
        "restart":          "Restart",
        "proposed_control": r"\textbf{Proposed}",
    }
    return pretty.get(name, name.replace("_", r"\_"))


def write_tables_from_summary(summary_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(summary_path).read_text())

    paths: dict[str, Path] = {}

    p = out_dir / "table_main.tex"
    p.write_text(main_results_table(summary))
    paths["main"] = p

    # Common NFEs to compare across samplers
    p = out_dir / "table_fid_at_nfe.tex"
    p.write_text(fid_at_matched_nfe_table(summary, nfes=[5, 8, 12, 18, 32, 64]))
    paths["fid_at_nfe"] = p

    p = out_dir / "table_pareto_auc.tex"
    p.write_text(pareto_auc_table(summary))
    paths["pareto_auc"] = p

    return paths
