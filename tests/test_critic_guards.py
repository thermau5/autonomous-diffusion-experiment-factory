import copy
import numpy as np
import pytest
import yaml
from pathlib import Path

from autonomous_diffusion.critic.guards import (
    ContractViolation,
    assert_locked_test_unchanged,
    check_seed_determinism,
    forbid_baseline_removal,
    forbid_best_seed_reporting,
    forbid_primary_metric_change,
    forbid_test_split_in_validation,
    freeze_contract,
)


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "benchmark_contract.yaml"


@pytest.fixture
def contract():
    return yaml.safe_load(CONTRACT_PATH.read_text())


def test_unchanged_contract_passes_locked_check(contract):
    rec = freeze_contract(contract)
    assert_locked_test_unchanged(contract, rec)  # no raise


def test_changed_primary_metric_aborts_locked_test(contract):
    rec = freeze_contract(contract)
    mutated = copy.deepcopy(contract)
    mutated["metrics"]["primary"] = ["clean_fid"]   # narrowed
    with pytest.raises(ContractViolation, match="drift"):
        assert_locked_test_unchanged(mutated, rec)


def test_added_baseline_after_freeze_aborts(contract):
    rec = freeze_contract(contract)
    mutated = copy.deepcopy(contract)
    mutated["baselines"].append({"id": "sneaky_new_baseline", "family": "x", "notes": ""})
    with pytest.raises(ContractViolation, match="drift"):
        assert_locked_test_unchanged(mutated, rec)


def test_forbid_test_split_in_validation_raises():
    with pytest.raises(ContractViolation):
        forbid_test_split_in_validation(phase="validation", split="test")
    forbid_test_split_in_validation(phase="validation", split="validation")
    forbid_test_split_in_validation(phase="test", split="test")


def test_forbid_baseline_removal_catches_drop(contract):
    declared = {b["id"] for b in contract["baselines"]}
    declared.discard("dpm_solver_pp")
    with pytest.raises(ContractViolation, match="silently dropped"):
        forbid_baseline_removal(contract, declared)


def test_forbid_best_seed_reporting():
    rec_ok = {
        "primary": {
            "clean_fid": {
                "mean": 2.5, "sem": 0.1,
                "per_seed": {"0": 2.4, "1": 2.6, "2": 2.5},
            },
        }
    }
    forbid_best_seed_reporting(rec_ok)

    rec_bad_single_seed = {"primary": {"clean_fid": {"mean": 2.5, "sem": 0.0, "per_seed": {"0": 2.5}}}}
    with pytest.raises(ContractViolation):
        forbid_best_seed_reporting(rec_bad_single_seed)

    rec_no_per_seed = {"primary": {"clean_fid": {"mean": 2.5, "sem": 0.1}}}
    with pytest.raises(ContractViolation):
        forbid_best_seed_reporting(rec_no_per_seed)


def test_forbid_primary_metric_change(contract):
    rec = freeze_contract(contract)
    forbid_primary_metric_change(contract, rec)
    mutated = copy.deepcopy(contract)
    mutated["metrics"]["primary"].append("dream_metric")
    with pytest.raises(ContractViolation):
        forbid_primary_metric_change(mutated, rec)


def test_determinism_check_bit_exact():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(4, 3, 8, 8), dtype=np.uint8)
    check_seed_determinism(a, a.copy(), sampler_id="x", seed=0)
    b = a.copy()
    b[0, 0, 0, 0] += 1
    with pytest.raises(ContractViolation, match="determinism"):
        check_seed_determinism(a, b, sampler_id="x", seed=0)
