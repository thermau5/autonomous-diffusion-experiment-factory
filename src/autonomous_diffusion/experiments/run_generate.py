"""CLI: generate samples with a registered sampler.

Writes:
  <run_root>/<run_id>/config_used.yaml
  <samples_root>/<run_id>/samples.npz       (uint8 BCHW)
  <run_root>/<run_id>/log.txt
  <run_root>/<run_id>/env.json

The eval step (run_eval) consumes (run_id, samples.npz) and writes metrics.json.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch

from ..critic.guards import forbid_test_split_in_validation
from ..models import load_edm_network
from ..samplers import get_sampler
from ._run_utils import (
    env_snapshot,
    find_dataset_cfg,
    load_contract,
    make_run_dir,
    make_run_id,
    stopwatch,
    write_json,
    write_yaml,
)


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"ad.gen.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


@click.command()
@click.option("--contract", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dataset", required=True, type=str)
@click.option("--sampler", "sampler_id", required=True, type=str)
@click.option("--nfe", required=True, type=int, help="Heun: NFE = 2*num_steps - 1; Euler: NFE = num_steps")
@click.option("--seed", required=True, type=int)
@click.option("--samples", "num_samples", required=True, type=int)
@click.option("--phase", required=True, type=click.Choice(["smoke", "validation", "test"]))
@click.option("--samples-root", default="outputs/samples", type=click.Path())
@click.option("--run-root",     default="outputs/runs",    type=click.Path())
@click.option("--batch-size", default=64, type=int)
@click.option("--device", default="cuda", type=str)
def main(
    contract: str,
    dataset: str,
    sampler_id: str,
    nfe: int,
    seed: int,
    num_samples: int,
    phase: str,
    samples_root: str,
    run_root: str,
    batch_size: int,
    device: str,
) -> None:
    contract_d = load_contract(contract)
    dataset_cfg = find_dataset_cfg(contract_d, dataset)

    # Even though run_generate is just a sampler driver, refuse silently-wrong
    # combinations: phase=validation must not be paired with a test-sized run.
    if phase == "validation":
        forbid_test_split_in_validation(phase=phase, split="validation")

    # Map NFE -> num_steps for EDM samplers.
    if sampler_id == "edm_heun":
        if nfe % 2 == 0:
            raise click.UsageError("edm_heun NFE must be odd (NFE = 2*num_steps - 1)")
        num_steps = (nfe + 1) // 2
    elif sampler_id == "edm_euler":
        num_steps = nfe
    else:
        # other samplers will define their own mapping; default 1:1
        num_steps = nfe

    run_id = make_run_id(dataset=dataset, sampler=sampler_id, nfe=nfe, seed=seed, phase=phase)
    run_dir = make_run_dir(run_root, run_id)
    sample_dir = Path(samples_root) / run_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(run_dir / "log.txt")
    log.info(f"run_id={run_id}")
    log.info(f"dataset={dataset} sampler={sampler_id} nfe={nfe} num_steps={num_steps} "
             f"seed={seed} samples={num_samples} batch={batch_size} device={device}")

    config_used = {
        "run_id":      run_id,
        "phase":       phase,
        "dataset":     dataset,
        "dataset_cfg": dataset_cfg,
        "sampler":     sampler_id,
        "nfe":         nfe,
        "num_steps":   num_steps,
        "seed":        seed,
        "num_samples": num_samples,
        "batch_size":  batch_size,
        "device":      device,
        "samples_root": str(samples_root),
        "run_root":    str(run_root),
        "contract_path": str(contract),
        "contract_version": contract_d.get("contract_version"),
        "sample_npz":  str(sample_dir / "samples.npz"),
    }
    write_yaml(run_dir / "config_used.yaml", config_used)
    write_json(run_dir / "env.json", env_snapshot())

    log.info("loading EDM network ...")
    net = load_edm_network(dataset, device=device)
    log.info(f"net loaded: img_resolution={getattr(net, 'img_resolution', '?')} "
             f"img_channels={getattr(net, 'img_channels', '?')} "
             f"sigma_min={getattr(net, 'sigma_min', '?')} sigma_max={getattr(net, 'sigma_max', '?')}")

    sampler = get_sampler(sampler_id)
    log.info(f"sampling {num_samples} images ...")
    with stopwatch() as sw, torch.inference_mode():
        out = sampler.sample(
            net=net,
            num_samples=num_samples,
            num_steps=num_steps,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )
    log.info(f"sampling done: {num_samples} images, NFE/sample={out.nfe}, wall={sw.elapsed:.2f}s")

    # uint8 BCHW for compact storage and FID input.
    x = out.samples.detach().float().clamp(-1, 1)
    x = ((x + 1) * 0.5 * 255.0 + 0.5).to(torch.uint8).numpy()
    np.savez_compressed(sample_dir / "samples.npz", samples=x, nfe=out.nfe)
    log.info(f"wrote samples to {sample_dir / 'samples.npz'} ({x.nbytes/1e6:.1f} MB)")

    run_summary = {
        "run_id":          run_id,
        "samples_path":    str(sample_dir / "samples.npz"),
        "nfe_per_sample":  int(out.nfe),
        "wall_seconds":    float(sw.elapsed),
        "metadata":        out.metadata,
    }
    write_json(run_dir / "generate_summary.json", run_summary)


if __name__ == "__main__":
    main()
