"""CLI: compute Clean-FID for an already-generated run.

Reads:  <run_root>/<run_id>/config_used.yaml, generate_summary.json
        + samples.npz from the configured samples path
Writes: <run_root>/<run_id>/metrics.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import yaml

from ..metrics import CleanFIDConfig, compute_clean_fid
from ._run_utils import find_dataset_cfg, find_latest_run, load_contract, write_json


def _logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    return log


RETENTION_CHOICES = ["keep_all", "seed0_only", "delete_all"]


def _apply_retention(sample_path: Path, *, seed: int, retention: str, log: logging.Logger) -> str:
    """Apply the retention policy AFTER FID is computed. Returns the action taken."""
    if retention == "keep_all":
        log.info(f"retention=keep_all: keeping {sample_path}")
        return "kept"
    if retention == "delete_all":
        if sample_path.exists():
            size = sample_path.stat().st_size
            sample_path.unlink()
            log.info(f"retention=delete_all: deleted {sample_path} ({size/1e6:.1f} MB)")
            return "deleted"
        return "missing"
    if retention == "seed0_only":
        if seed == 0:
            log.info(f"retention=seed0_only: keeping seed-0 samples at {sample_path}")
            return "kept_seed0"
        if sample_path.exists():
            size = sample_path.stat().st_size
            sample_path.unlink()
            log.info(f"retention=seed0_only: deleted non-seed-0 samples at {sample_path} ({size/1e6:.1f} MB)")
            return "deleted_non_seed0"
        return "missing"
    raise click.UsageError(f"unknown retention policy {retention!r}")


@click.command()
@click.option("--contract", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dataset", required=True, type=str)
@click.option("--phase", required=True, type=click.Choice(["smoke", "validation", "test"]))
@click.option("--run-root", default="outputs/runs", type=click.Path())
@click.option("--run-id", default=None, type=str, help="Specific run id; otherwise --latest")
@click.option("--latest", is_flag=True, help="Eval the most recent matching run")
@click.option("--fid-mode", default="clean", type=click.Choice(["clean", "legacy_pytorch", "legacy_tensorflow"]))
@click.option("--retention", default="seed0_only", type=click.Choice(RETENTION_CHOICES),
              help="What to do with samples.npz after FID is computed.")
def main(
    contract: str,
    dataset: str,
    phase: str,
    run_root: str,
    run_id: str | None,
    latest: bool,
    fid_mode: str,
    retention: str,
) -> None:
    log = _logger("ad.eval")
    contract_d = load_contract(contract)
    dataset_cfg = find_dataset_cfg(contract_d, dataset)

    if run_id is None and not latest:
        raise click.UsageError("provide --run-id or --latest")
    if run_id is not None:
        run_dir = Path(run_root) / run_id
        if not run_dir.is_dir():
            raise click.UsageError(f"run dir not found: {run_dir}")
    else:
        run_dir = find_latest_run(run_root, dataset=dataset, phase=phase)

    cfg_used = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    sample_path = Path(cfg_used["sample_npz"])
    if not sample_path.exists():
        raise FileNotFoundError(f"samples not found at {sample_path}")

    log.info(f"evaluating run {run_dir.name}")
    log.info(f"loading {sample_path} ...")
    arr = np.load(sample_path)["samples"]   # uint8 BCHW
    log.info(f"loaded {arr.shape} dtype={arr.dtype}")

    fid_cfg = CleanFIDConfig(
        dataset_name=dataset,
        dataset_res=int(dataset_cfg["resolution"]),
        dataset_split=str(dataset_cfg.get("fid_reference_split", "train")),
        mode=fid_mode,
    )
    log.info(f"computing Clean-FID against {fid_cfg.dataset_name}-{fid_cfg.dataset_res} "
             f"split={fid_cfg.dataset_split} mode={fid_cfg.mode} ...")
    fid_out = compute_clean_fid(arr, fid_cfg)
    log.info(f"FID = {fid_out['fid']:.4f}  (n={fid_out['n_samples']})")

    gen_summary = {}
    gs_path = run_dir / "generate_summary.json"
    if gs_path.exists():
        import json
        gen_summary = json.loads(gs_path.read_text())

    retention_action = _apply_retention(
        sample_path, seed=int(cfg_used["seed"]), retention=retention, log=log,
    )

    metrics = {
        "run_id":         cfg_used["run_id"],
        "phase":          phase,
        "dataset":        dataset,
        "sampler":        cfg_used["sampler"],
        "nfe":            int(cfg_used["nfe"]),
        "seed":           int(cfg_used["seed"]),
        "num_samples":    int(cfg_used["num_samples"]),
        "wall_seconds":   gen_summary.get("wall_seconds"),
        "nfe_per_sample": gen_summary.get("nfe_per_sample"),
        "clean_fid":      fid_out["fid"],
        "fid_mode":       fid_out["mode"],
        "fid_ref_split":  fid_out["split"],
        "retention":      {"policy": retention, "action": retention_action,
                           "sample_path": str(sample_path) if retention_action in ("kept","kept_seed0") else None},
    }
    write_json(run_dir / "metrics.json", metrics)
    log.info(f"wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
