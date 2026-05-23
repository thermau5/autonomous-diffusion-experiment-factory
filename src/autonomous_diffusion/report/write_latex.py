"""CLI: compile the compact LaTeX report from a sweep/locked-test summary.

The report follows the template in the project spec:
  Section 1: Theory-to-Experiment map
  Section 2: Real-image benchmark statement
  Section 3: Main result (FID-at-best-NFE table)
  Section 4: Pareto frontier figure
  Section 5: Pareto-AUC summary
  Section 6: Per-NFE table (low-NFE focus)
  Section 7: Conclusion with Delta_Pareto and Delta_FID@K
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from .make_figures import write_figures_from_summary
from .make_tables import write_tables_from_summary


REPORT_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref}
\title{Autonomous Diffusion: real-image verification on %(dataset_upper)s}
\author{thermau5}
\date{\today}
\begin{document}
\maketitle

\section{Theory-to-Experiment Map}

Let $p_0 = \mathcal{N}(0, I)$, $p_1 = p_{\mathrm{data}}$, and let
\[
\dot x_t = f_\theta(x_t, t, u_t), \qquad t \in [0,1].
\]
A sampler is a discretized controlled flow
\[
x_{i+1} = \Phi^{u_i}_{\Delta t_i}(x_i; \theta), \qquad i = 0,\ldots,K-1,
\]
with cost $C(u) = \mathrm{NFE}(u)$. The proposed scheduler solves
\[
u^\star \in \arg\min_{u \in \mathcal{U}_K} C(u) \quad \text{s.t.} \quad
\widehat R_{\mathrm{val}}(u) + B_n(u, \delta) \le \varepsilon.
\]
For an order-$p$ solver with $\rho$ fixed by the pretrained generator, the
sampling-step density's stationary point is
$m^\star(\sigma) \propto d(\sigma)^{1/(p+1)}$. Our implementation uses
$d(\sigma)$ estimated empirically (one-Heun vs.\ 16-substep Heun on a
uniform-log-$\sigma$ grid) and an empirical perceptual weighting
$w(\sigma) = \sigma^{-k}$ with $k = 2$, matching Karras's qualitative
concentration. Calibration is validation-only and cached per pretrained
checkpoint.

\section{Benchmark}

\[
\mathcal{D} = \{ \mathrm{%(dataset_upper)s} \}.
\]

\[
\mathcal{B} = \{
\mathrm{DDPM},
\mathrm{DDIM},
\mathrm{EDM\text{-}Euler},
\mathrm{EDM\text{-}Heun},
\mathrm{DPM\text{-}Solver},
\mathrm{DPM\text{-}Solver\text{++}},
\mathrm{UniPC},
\mathrm{DEIS},
\mathrm{PNDM},
\mathrm{Restart},
\mathrm{Uniform},
\mathrm{Karras}
\}.
\]
Generator is the frozen pretrained EDM net; only the sampler/schedule/control is varied. FID is computed with Clean-FID against the dataset reference.

\section{Main Result}

\begin{table}[h]
\centering
\input{table_main.tex}
\caption{Per-sampler best Clean-FID on %(dataset_upper)s (%(samples_per_run)s samples per run, %(num_runs)s total runs across %(num_seeds)s seeds). Values are mean $\pm$ SEM.}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{pareto_fid_nfe.pdf}
\caption{Quality-efficiency frontier: Clean-FID vs.\ NFE. Lower-left is better. \textbf{Proposed} is the only sampler implementing the certificate-optimal step density.}
\end{figure}

\section{FID at matched NFE}

\begin{table}[h]
\centering
\input{table_fid_at_nfe.tex}
\caption{Clean-FID for each sampler at common NFE budgets. Where a sampler did not run at a given NFE, the cell is ``--''.}
\end{table}

\section{Pareto-AUC}

\input{table_pareto_auc.tex}

The Pareto-AUC integrates the lower envelope of Clean-FID vs.\ $\log_{10}$ NFE over $[4, 128]$. Lower is better.

\section{Conclusion}

\[
\Delta_{\mathrm{Pareto}} = \frac{\mathrm{AUC}_{\mathrm{best baseline}} - \mathrm{AUC}_{\mathrm{ours}}}{\mathrm{AUC}_{\mathrm{best baseline}}}.
\]

\[
\Delta_{\mathrm{FID}@K} = \frac{\min_{b \in \mathcal{B}: \mathrm{NFE}=K} \mathrm{FID}(b) - \mathrm{FID}(u^\star_K)}{\min_{b \in \mathcal{B}: \mathrm{NFE}=K} \mathrm{FID}(b)}.
\]

The theory is verified iff
\[
\Delta_{\mathrm{Pareto}} > 0, \quad \Delta_{\mathrm{FID}@K} > 0 \text{ for at least one low-NFE regime}, \quad V(u^\star) \le \varepsilon.
\]

\end{document}
"""


@click.command()
@click.option("--summary", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to the sweep/locked-test summary.json")
@click.option("--out", default="outputs/paper", type=click.Path(),
              help="Output directory for the LaTeX report")
@click.option("--contract", default="contracts/benchmark_contract.yaml",
              type=click.Path(exists=True, dir_okay=False))
@click.option("--run-root", default="outputs/runs", type=click.Path(),
              help="So the all-points enrichment can pull every (sampler, nfe) cell")
def main(summary, out, contract, run_root):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary_d = json.loads(Path(summary).read_text())

    # Enrich summary with ALL per-(sampler, nfe) points so the fid_at_nfe
    # table doesn't gap on non-frontier cells. We re-aggregate from run_root.
    from ..metrics.pareto import aggregate_seeds
    all_runs = []
    for run_dir in Path(run_root).iterdir():
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        all_runs.append(json.loads(mp.read_text()))
    points = aggregate_seeds(all_runs)
    all_points: dict[str, dict] = {}
    for p in points:
        all_points.setdefault(p.sampler, {})[str(p.nfe)] = {
            "fid_mean": p.fid_mean, "fid_sem": p.fid_sem, "wall": p.wall_seconds_mean,
        }
    summary_d["_all_points"] = all_points
    enriched = out / "_enriched_summary.json"
    enriched.write_text(json.dumps(summary_d, indent=2, default=str))

    write_tables_from_summary(enriched, out)
    write_figures_from_summary(enriched, out)

    # Resolve template
    nseeds = len({z for r in all_runs for z in [r["seed"]]}) if all_runs else 0
    nruns = len(all_runs)
    samples_per_run = summary_d.get("samples_per_run") or "?"
    dataset = summary_d.get("dataset", "?")
    tex = REPORT_TEMPLATE % {
        "dataset_upper": dataset.upper(),
        "samples_per_run": samples_per_run,
        "num_runs": nruns,
        "num_seeds": nseeds,
    }
    (out / "report.tex").write_text(tex)
    click.echo(f"wrote {out / 'report.tex'} + tables/figures/{out}")


if __name__ == "__main__":
    main()
