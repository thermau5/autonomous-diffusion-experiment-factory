"""CLI: run a validation sweep over (sampler, NFE, seed).

This driver IS allowed to tune. It writes per-run metrics.json files via
the standard run_generate + run_eval path, plus a sweep-level summary
under outputs/sweeps/<sweep_id>/.

It is NOT the locked-test runner -- the contract's `locked_test_freeze`
guards are not invoked here. They are invoked by run_locked_test.py.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import click
import yaml

from ..critic.guards import forbid_test_split_in_validation
from ..metrics.pareto import aggregate_seeds, pareto_auc, pareto_frontier, per_sampler_summary
from ._run_utils import env_snapshot, load_contract, write_json


def _log() -> logging.Logger:
    log = logging.getLogger("ad.sweep")
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    return log


def _run(cmd: list[str], log: logging.Logger) -> int:
    log.info("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error("FAILED (rc=%d): %s", p.returncode, p.stderr.strip().splitlines()[-3:])
    return p.returncode


def _nfe_for(sampler: str, nfe_in: int) -> int:
    """Adjust to the actual NFE the sampler will run at, matching run_generate.
    Heun-class samplers need ODD NFE for the 2*K-1 convention; for them we
    round nfe_in up to the next odd number."""
    two_call = {"edm_heun", "dpm_solver"}
    if sampler in two_call and nfe_in % 2 == 0:
        return nfe_in + 1
    return nfe_in


@click.command()
@click.option("--contract", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dataset", required=True, type=str)
@click.option("--phase", default="validation", type=click.Choice(["validation", "smoke"]))
@click.option("--samples", default=None, type=int, help="Override per-run sample count. Defaults to dataset.samples_validation (smoke uses samples_smoke).")
@click.option("--nfe-grid", default=None, type=str, help="Comma-separated NFE list; defaults to budgets.nfe_grid from contract")
@click.option("--seeds", default=None, type=str, help="Comma-separated seed list; defaults to budgets.seeds from contract")
@click.option("--samplers", default=None, type=str, help="Comma-separated sampler ids; defaults to all baselines + proposed_control")
@click.option("--hours", default=10.0, type=float, help="Wall-clock time budget; sweep stops after this even if grid not exhausted")
@click.option("--samples-root", default="outputs/samples", type=click.Path())
@click.option("--run-root", default="outputs/runs", type=click.Path())
@click.option("--sweep-root", default="outputs/sweeps", type=click.Path())
@click.option("--retention", default="seed0_only", type=click.Choice(["keep_all", "seed0_only", "delete_all"]))
@click.option("--batch-size", default=64, type=int)
@click.option("--device", default="cuda")
def main(
    contract, dataset, phase, samples, nfe_grid, seeds, samplers, hours,
    samples_root, run_root, sweep_root, retention, batch_size, device,
):
    log = _log()
    contract_d = load_contract(contract)
    forbid_test_split_in_validation(phase=phase, split=phase)

    dataset_cfg = next(
        d for bucket in ("primary", "secondary")
        for d in (contract_d.get("datasets", {}).get(bucket) or [])
        if d["name"] == dataset
    )

    if samples is None:
        samples = (
            dataset_cfg["samples_smoke"] if phase == "smoke"
            else dataset_cfg["samples_validation"]
        )
    if nfe_grid is None:
        nfes = list(contract_d["budgets"]["nfe_grid"])
    else:
        nfes = [int(x) for x in nfe_grid.split(",")]
    if seeds is None:
        seed_list = list(contract_d["budgets"]["seeds"])
    else:
        seed_list = [int(x) for x in seeds.split(",")]
    if samplers is None:
        sampler_list = [b["id"] for b in contract_d["baselines"]] + [contract_d["proposed"]["name"].replace(" ", "_")]
        # contract proposed name is "risk_constrained_control_scheduler", but the
        # registered sampler id is "proposed_control". Use the registered id.
        sampler_list = [b["id"] for b in contract_d["baselines"]] + ["proposed_control"]
    else:
        sampler_list = [s.strip() for s in samplers.split(",")]

    sweep_id = time.strftime("%Y%m%dT%H%M%S") + f"_{dataset}_{phase}"
    sweep_dir = Path(sweep_root) / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for sampler in sampler_list:
        for nfe_in in nfes:
            nfe_actual = _nfe_for(sampler, nfe_in)
            for seed in seed_list:
                plan.append({"sampler": sampler, "nfe": nfe_actual, "seed": seed})

    write_json(sweep_dir / "plan.json", {
        "dataset": dataset, "phase": phase, "samples_per_run": samples,
        "nfe_grid": nfes, "seeds": seed_list, "samplers": sampler_list,
        "num_runs": len(plan), "hours_budget": hours,
        "contract_path": str(contract),
        "contract_version": contract_d.get("contract_version"),
        "env": env_snapshot(),
    })
    log.info(f"sweep_id={sweep_id}  plan={len(plan)} runs  samples/run={samples}  budget={hours}h")

    t0 = time.time()
    completed = []
    failed = []
    py = [sys.executable, "-m"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))}

    for i, p in enumerate(plan):
        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h >= hours:
            log.warning(f"hour budget exhausted ({elapsed_h:.2f}h >= {hours}h); stopping sweep at run {i}/{len(plan)}")
            break
        log.info(f"[{i+1:>4d}/{len(plan)}] {p['sampler']} nfe={p['nfe']} seed={p['seed']}  elapsed={elapsed_h:.2f}h")
        gen_rc = _run(py + [
            "autonomous_diffusion.experiments.run_generate",
            "--contract", contract,
            "--dataset", dataset,
            "--sampler", p["sampler"],
            "--nfe", str(p["nfe"]),
            "--seed", str(p["seed"]),
            "--samples", str(samples),
            "--phase", phase,
            "--samples-root", samples_root,
            "--run-root", run_root,
            "--batch-size", str(batch_size),
            "--device", device,
        ], log)
        if gen_rc != 0:
            failed.append({"plan_index": i, **p, "stage": "generate"})
            continue
        eval_rc = _run(py + [
            "autonomous_diffusion.experiments.run_eval",
            "--contract", contract,
            "--dataset", dataset,
            "--phase", phase,
            "--latest",
            "--retention", retention,
        ], log)
        if eval_rc != 0:
            failed.append({"plan_index": i, **p, "stage": "eval"})
            continue
        completed.append({"plan_index": i, **p})

    # Aggregate
    log.info(f"sweep done: {len(completed)} ok, {len(failed)} failed")
    runs = []
    for run_dir in Path(run_root).iterdir():
        if f"_{phase}_{dataset}_" not in run_dir.name:
            continue
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        runs.append(json.loads(mp.read_text()))
    log.info(f"aggregating {len(runs)} metrics files")

    points = aggregate_seeds(runs)
    summary = {
        "sweep_id": sweep_id,
        "dataset": dataset, "phase": phase, "samples_per_run": samples,
        "num_runs_completed": len(completed),
        "num_runs_failed": len(failed),
        "failed": failed,
        "per_sampler": per_sampler_summary(points),
        "frontier": [
            {"sampler": p.sampler, "nfe": p.nfe, "fid_mean": p.fid_mean,
             "fid_sem": p.fid_sem, "wall_seconds_mean": p.wall_seconds_mean}
            for p in pareto_frontier(points)
        ],
        "pareto_auc_fid_nfe_log": pareto_auc(
            pareto_frontier(points), nfe_lo=4, nfe_hi=128,
        ),
    }
    write_json(sweep_dir / "summary.json", summary)
    log.info(f"wrote {sweep_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
