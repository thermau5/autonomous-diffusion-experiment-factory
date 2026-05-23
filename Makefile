SHELL := /bin/bash

CONDA_ENV       ?= autodiff
CONTRACT        ?= contracts/benchmark_contract.yaml
DATASET         ?= cifar10
SAMPLER         ?= edm_heun
NFE             ?= 18
SEED            ?= 0
SAMPLES         ?= 1024
SAMPLES_ROOT    ?= outputs/samples
RUN_ROOT        ?= outputs/runs
HOURS           ?= 10
GPU             ?= 0
RETENTION       ?= seed0_only       # keep_all | seed0_only | delete_all

PY := CUDA_VISIBLE_DEVICES=$(GPU) python -m

.PHONY: help setup vendor_edm test smoke validation_sweep select_validated_config \
        freeze_contract locked_test report clean autonomous_real_image_run

help:
	@echo "Targets:"
	@echo "  setup                       - create conda env, pip install -e ., vendor EDM"
	@echo "  vendor_edm                  - clone NVlabs/edm into third_party/edm"
	@echo "  test                        - pytest"
	@echo "  smoke                       - 1k sample smoke run on \$$DATASET / \$$SAMPLER / NFE=\$$NFE"
	@echo "  validation_sweep            - full validation sweep within \$$HOURS hours"
	@echo "  select_validated_config     - pick best validated config (writes freeze record)"
	@echo "  freeze_contract             - lock contract for the locked test"
	@echo "  locked_test                 - 50k locked-test eval. REFUSES if freeze drifts."
	@echo "  report                      - compile LaTeX report"

setup: vendor_edm
	@if ! conda env list | grep -q "^$(CONDA_ENV) "; then \
		echo ">> creating conda env $(CONDA_ENV)"; \
		conda env create -f environment.yml -n $(CONDA_ENV); \
	fi
	conda run -n $(CONDA_ENV) pip install -e .

vendor_edm:
	@if [ ! -d third_party/edm/.git ]; then \
		echo ">> cloning NVlabs/edm into third_party/edm"; \
		git clone --depth 1 https://github.com/NVlabs/edm.git third_party/edm; \
	else \
		echo ">> third_party/edm already present"; \
	fi

test:
	$(PY) pytest

smoke:
	$(PY) autonomous_diffusion.experiments.run_generate \
		--contract $(CONTRACT) \
		--dataset $(DATASET) \
		--sampler $(SAMPLER) \
		--nfe $(NFE) \
		--seed $(SEED) \
		--samples $(SAMPLES) \
		--phase smoke \
		--samples-root $(SAMPLES_ROOT) \
		--run-root $(RUN_ROOT)
	$(PY) autonomous_diffusion.experiments.run_eval \
		--contract $(CONTRACT) \
		--dataset $(DATASET) \
		--phase smoke \
		--latest \
		--retention $(RETENTION)

validation_sweep:
	$(PY) autonomous_diffusion.experiments.run_sweep \
		--contract $(CONTRACT) \
		--dataset $(DATASET) \
		--phase validation \
		--hours $(HOURS) \
		--samples-root $(SAMPLES_ROOT) \
		--run-root $(RUN_ROOT)

select_validated_config:
	$(PY) autonomous_diffusion.experiments.run_sweep \
		--contract $(CONTRACT) \
		--dataset $(DATASET) \
		--select-best \
		--write-freeze-record outputs/freeze_record.yaml

freeze_contract:
	$(PY) autonomous_diffusion.critic.freeze \
		--contract $(CONTRACT) \
		--out outputs/freeze_record.yaml

locked_test:
	$(PY) autonomous_diffusion.experiments.run_locked_test \
		--contract $(CONTRACT) \
		--freeze-record outputs/freeze_record.yaml \
		--dataset $(DATASET) \
		--samples-root $(SAMPLES_ROOT) \
		--run-root $(RUN_ROOT) \
		--hours $$(($(HOURS) * 2))

report:
	$(PY) autonomous_diffusion.report.write_latex \
		--contract $(CONTRACT) \
		--run-root $(RUN_ROOT) \
		--out outputs/paper

autonomous_real_image_run: setup test
	$(MAKE) smoke
	$(MAKE) validation_sweep
	$(MAKE) select_validated_config
	$(MAKE) freeze_contract
	$(MAKE) locked_test
	$(MAKE) report

clean:
	rm -rf outputs/runs/* outputs/samples/* outputs/paper/*
