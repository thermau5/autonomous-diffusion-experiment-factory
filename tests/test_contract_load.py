from pathlib import Path

import yaml


CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "benchmark_contract.yaml"


def test_contract_parses():
    data = yaml.safe_load(CONTRACT.read_text())
    assert data["project"] == "autonomous_diffusion_real_image_verification"
    assert "locked_test_freeze" in data and isinstance(data["locked_test_freeze"], list)


def test_contract_has_all_required_baselines():
    data = yaml.safe_load(CONTRACT.read_text())
    declared = {b["id"] for b in data["baselines"]}
    required = {
        "edm_euler", "edm_heun", "ddim", "ddpm_ancestral",
        "dpm_solver", "dpm_solver_pp", "unipc", "deis", "pndm",
        "restart", "uniform_schedule", "karras_schedule",
    }
    missing = required - declared
    assert not missing, f"contract missing baselines: {missing}"


def test_contract_forbids_test_tuning():
    data = yaml.safe_load(CONTRACT.read_text())
    assert "tune_on_locked_test" in data["forbidden"]
    assert "change_primary_metric_after_results" in data["forbidden"]
    assert "report_best_seed_only" in data["forbidden"]


def test_freeze_fields_resolve():
    from autonomous_diffusion.critic.guards import freeze_contract
    data = yaml.safe_load(CONTRACT.read_text())
    rec = freeze_contract(data)
    assert "sha256" in rec and len(rec["sha256"]) == 64
    assert rec["snapshot"]  # non-empty
