"""Runtime guards that enforce the no-run-until-success contract.

Lint cannot see "we ran the test 7 times and picked the best." These asserts can.
Every guard fails closed: if the situation is ambiguous, raise ContractViolation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class ContractViolation(RuntimeError):
    """Raised when the experiment contract is violated at runtime."""


# ---------------------------------------------------------------------------
# Freeze record I/O
# ---------------------------------------------------------------------------

def _select(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ContractViolation(f"freeze field {dotted!r} not present in contract")
        cur = cur[part]
    return cur


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def freeze_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot the locked_test_freeze fields of a contract.

    Returns a record carrying both the resolved values and their hash, so a
    later run can diff them deterministically.
    """
    freeze_fields = contract.get("locked_test_freeze") or []
    if not freeze_fields:
        raise ContractViolation("contract has no `locked_test_freeze:` field list")
    snapshot = {f: _select(contract, f) for f in freeze_fields}
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    return {
        "contract_version": contract.get("contract_version"),
        "project": contract.get("project"),
        "frozen_fields": freeze_fields,
        "snapshot": snapshot,
        "sha256": digest,
    }


def load_freeze_record(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# The actual guards
# ---------------------------------------------------------------------------

def assert_locked_test_unchanged(
    contract: Mapping[str, Any], freeze_record: Mapping[str, Any]
) -> None:
    """Refuse to run the locked test if any frozen field has drifted."""
    if contract.get("project") != freeze_record.get("project"):
        raise ContractViolation(
            f"project mismatch: contract={contract.get('project')!r} "
            f"freeze={freeze_record.get('project')!r}"
        )
    current = freeze_contract(contract)
    if current["sha256"] != freeze_record["sha256"]:
        diffs = []
        for f in current["frozen_fields"]:
            now = current["snapshot"].get(f)
            then = freeze_record["snapshot"].get(f)
            if _canonical(now) != _canonical(then):
                diffs.append(f)
        raise ContractViolation(
            "locked_test_freeze drift detected; the following fields changed since "
            f"freeze: {diffs}. The locked test cannot be run on a mutated contract."
        )


def forbid_test_split_in_validation(*, phase: str, split: str) -> None:
    """Any phase=validation code path that touches split=test is a bug."""
    if phase == "validation" and split == "test":
        raise ContractViolation(
            "validation phase attempted to load the test split; this is forbidden."
        )


def forbid_baseline_removal(
    contract: Mapping[str, Any], run_plan_baselines: Iterable[str]
) -> None:
    """Every baseline in the contract must appear in the run plan."""
    declared = {b["id"] for b in contract.get("baselines", [])}
    planned = set(run_plan_baselines)
    missing = declared - planned
    if missing:
        raise ContractViolation(
            f"baselines silently dropped from the run plan: {sorted(missing)}. "
            "Removing a baseline post-failure is forbidden."
        )


def forbid_primary_metric_change(
    contract: Mapping[str, Any], freeze_record: Mapping[str, Any]
) -> None:
    cur = list(contract.get("metrics", {}).get("primary", []))
    then = list(freeze_record["snapshot"].get("metrics.primary", []))
    if cur != then:
        raise ContractViolation(
            f"primary metric set changed since freeze. "
            f"frozen={then!r} current={cur!r}"
        )


def forbid_best_seed_reporting(metrics_record: Mapping[str, Any]) -> None:
    """A primary number must carry per-seed values and aggregate as mean+SEM."""
    primary = metrics_record.get("primary")
    if primary is None:
        raise ContractViolation("metrics_record has no `primary` block")
    for name, payload in primary.items():
        if "per_seed" not in payload:
            raise ContractViolation(
                f"primary metric {name!r} reported without per_seed values"
            )
        if "mean" not in payload or "sem" not in payload:
            raise ContractViolation(
                f"primary metric {name!r} reported without mean+sem aggregation"
            )
        if len(payload["per_seed"]) < 2:
            raise ContractViolation(
                f"primary metric {name!r} reported with <2 seeds (=best-seed style)"
            )


def check_seed_determinism(
    samples_a, samples_b, *, sampler_id: str, seed: int, atol: float = 0.0
) -> None:
    """Same seed -> identical samples. Atol defaults to 0 (bit-exact)."""
    import numpy as np
    a = np.asarray(samples_a)
    b = np.asarray(samples_b)
    if a.shape != b.shape:
        raise ContractViolation(
            f"determinism check: shape mismatch {a.shape} vs {b.shape} "
            f"for sampler={sampler_id!r} seed={seed}"
        )
    diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    if diff > atol:
        raise ContractViolation(
            f"determinism check failed: max abs diff = {diff:g} > atol={atol:g} "
            f"for sampler={sampler_id!r} seed={seed}"
        )
