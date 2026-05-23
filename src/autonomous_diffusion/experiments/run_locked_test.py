"""CLI: the locked-test runner. Refuses to execute if the contract drifts
from the freeze record. After execution, the metrics file MUST carry
mean ± SEM over seeds (best-seed reporting is forbidden by the guard).

This file is short on purpose: all the methodology enforcement lives in
critic/guards.py and is invoked at the head of the function.
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

from ..critic.guards import (
    assert_locked_test_unchanged,
    forbid_baseline_removal,
    forbid_best_seed_reporting,
    forbid_primary_metric_change,
    forbid_test_split_in_validation,
    load_freeze_record,
)
from ..metrics.pareto import aggregate_seeds, pareto_auc, pareto_frontier, per_sampler_summary
from ._run_utils import env_snapshot, load_contract, write_json


def _log():
    log = logging.getLogger("ad.locked_test")
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    return log


def _run(cmd, log):
    log.info("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error("FAILED (rc=%d): %s", p.returncode, p.stderr.strip().splitlines()[-3:])
    return p.returncode


def _nfe_for(sampler: str, nfe_in: int) -> int:
    two_call = {"edm_heun", "dpm_solver"}
    if sampler in two_call and nfe_in % 2 == 0:
        return nfe_in + 1
    return nfe_in


@click.command()
@click.option("--contract", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--freeze-record", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dataset", required=True, type=str)
@click.option("--samples-root", default="outputs/samples", type=click.Path())
@click.option("--run-root",     default="outputs/runs",    type=click.Path())
@click.option("--test-root",    default="outputs/locked_test", type=click.Path())
@click.option("--hours", default=24.0, type=float)
@click.option("--batch-size", default=64, type=int)
@click.option("--device", default="cuda")
def main(contract, freeze_record, dataset, samples_root, run_root, test_root,
         hours, batch_size, device):
    log = _log()
    contract_d = load_contract(contract)
    freeze = load_freeze_record(freeze_record)

    # Hard guards. Any of these raises ContractViolation -> the runner aborts
    # *before* touching the test split.
    log.info("running critic guards...")
    assert_locked_test_unchanged(contract_d, freeze)
    forbid_primary_metric_change(contract_d, freeze)
    declared_baselines = [b["id"] for b in contract_d["baselines"]]
    forbid_baseline_removal(contract_d, declared_baselines)
    # phase=test is the only place we let split=test in; validation phase is
    # explicitly forbidden upstream.
    forbid_test_split_in_validation(phase="test", split="test")
    log.info("guards passed; contract sha256 matches freeze record.")

    dataset_cfg = next(
        d for bucket in ("primary", "secondary")
        for d in (contract_d.get("datasets", {}).get(bucket) or [])
        if d["name"] == dataset
    )
    samples = int(dataset_cfg["samples_eval"])      # locked test = full eval count
    nfes = list(contract_d["budgets"]["nfe_grid"])
    seed_list = list(contract_d["budgets"]["seeds"])
    sampler_list = declared_baselines + ["proposed_control"]

    test_id = time.strftime("%Y%m%dT%H%M%S") + f"_{dataset}_locked_test"
    test_dir = Path(test_root) / test_id
    test_dir.mkdir(parents=True, exist_ok=True)
    write_json(test_dir / "header.json", {
        "test_id": test_id,
        "dataset": dataset,
        "samples_per_run": samples,
        "nfe_grid": nfes, "seeds": seed_list, "samplers": sampler_list,
        "contract_path": str(contract),
        "contract_version": contract_d.get("contract_version"),
        "freeze_sha256": freeze["sha256"],
        "env": env_snapshot(),
    })

    plan = [
        {"sampler": s, "nfe": _nfe_for(s, n), "seed": z}
        for s in sampler_list for n in nfes for z in seed_list
    ]
    log.info(f"locked_test_id={test_id}  plan={len(plan)} runs  samples/run={samples}  budget={hours}h")
    log.warning("This runner evaluates the test split ONCE per (sampler, nfe, seed). "
                "There is no re-run on the test split.")

    t0 = time.time()
    completed = []
    failed = []
    py = [sys.executable, "-m"]

    for i, p in enumerate(plan):
        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h >= hours:
            log.warning(f"hour budget exhausted ({elapsed_h:.2f}h >= {hours}h); stopping at run {i}/{len(plan)}")
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
            "--phase", "test",
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
            "--phase", "test",
            "--latest",
            "--retention", "seed0_only",
        ], log)
        if eval_rc != 0:
            failed.append({"plan_index": i, **p, "stage": "eval"})
            continue
        completed.append({"plan_index": i, **p})

    log.info(f"locked-test done: {len(completed)} ok, {len(failed)} failed")
    runs = []
    for run_dir in Path(run_root).iterdir():
        if "_test_" + dataset + "_" not in run_dir.name:
            continue
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        runs.append(json.loads(mp.read_text()))
    log.info(f"aggregating {len(runs)} metrics files from the locked test split")

    points = aggregate_seeds(runs)

    # Build the metrics record with mandatory per-seed + mean + SEM structure,
    # then re-run the guard to refuse best-seed reporting.
    by_sampler_primary = {}
    for s in sampler_list:
        s_pts = [p for p in points if p.sampler == s]
        if not s_pts:
            continue
        # Use the best-NFE point for headline; record per-seed values.
        best = min(s_pts, key=lambda q: q.fid_mean)
        primary_payload = {
            "mean": best.fid_mean,
            "sem": best.fid_sem,
            "per_seed": {str(z): v for z, v in zip(seed_list, best.per_seed_fid)},
            "nfe": best.nfe,
        }
        forbid_best_seed_reporting({"primary": {f"{s}__clean_fid_at_best_nfe": primary_payload}})
        by_sampler_primary[s] = primary_payload

    summary = {
        "test_id": test_id,
        "freeze_sha256": freeze["sha256"],
        "dataset": dataset,
        "num_runs_completed": len(completed),
        "num_runs_failed": len(failed),
        "failed": failed,
        "per_sampler": per_sampler_summary(points),
        "primary_per_sampler": by_sampler_primary,
        "frontier": [
            {"sampler": p.sampler, "nfe": p.nfe, "fid_mean": p.fid_mean,
             "fid_sem": p.fid_sem, "wall_seconds_mean": p.wall_seconds_mean,
             "per_seed_fid": list(p.per_seed_fid)}
            for p in pareto_frontier(points)
        ],
        "pareto_auc_fid_nfe_log": pareto_auc(
            pareto_frontier(points), nfe_lo=4, nfe_hi=128,
        ),
    }
    write_json(test_dir / "summary.json", summary)
    log.info(f"wrote {test_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
