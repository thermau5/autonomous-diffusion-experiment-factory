"""Shared utilities for run scripts (run_generate / run_eval / run_sweep)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


def load_contract(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def make_run_id(*, dataset: str, sampler: str, nfe: int, seed: int, phase: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}_{phase}_{dataset}_{sampler}_nfe{nfe}_seed{seed}_{suffix}"


def make_run_dir(run_root: str | Path, run_id: str) -> Path:
    p = Path(run_root) / run_id
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_yaml(path: str | Path, data: Any) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=False, default=str)


def env_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "pid": os.getpid(),
        "env": {
            k: os.environ[k]
            for k in ["CUDA_VISIBLE_DEVICES", "PYTHONPATH"]
            if k in os.environ
        },
    }


def find_dataset_cfg(contract: dict[str, Any], name: str) -> dict[str, Any]:
    for bucket in ("primary", "secondary"):
        for d in contract.get("datasets", {}).get(bucket, []) or []:
            if d["name"] == name:
                return d
    raise KeyError(f"dataset {name!r} not in contract")


def find_latest_run(run_root: str | Path, *, dataset: str, phase: str) -> Path:
    root = Path(run_root)
    cands = [p for p in root.iterdir() if p.is_dir() and f"_{phase}_{dataset}_" in p.name]
    if not cands:
        raise FileNotFoundError(f"no runs found under {root} for dataset={dataset} phase={phase}")
    return sorted(cands)[-1]


@contextmanager
def stopwatch():
    """`with stopwatch() as sw: ...; sw.elapsed` after the block."""
    class _SW:
        elapsed: float = 0.0
    sw = _SW()
    t0 = time.perf_counter()
    try:
        yield sw
    finally:
        sw.elapsed = time.perf_counter() - t0
