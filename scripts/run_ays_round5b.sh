#!/usr/bin/env bash
# Round 5b: faithful AYS reimplementation on the UniPC core.
# Single-shot evaluation on the locked test split with the same
# NFE grid and seed grid as the locked test, so the result is directly
# comparable to the locked-test (Karras, UniPC) and (Ours, UniPC) rows.
#
# Outputs land in outputs/runs/ with run_ids tagged ays_unipc; the
# summary is aggregated by scripts/aggregate_ays_round5b.py afterwards.

set -euo pipefail

CONTRACT=contracts/benchmark_contract.yaml
DATASET=cifar10
SAMPLES=10000
PHASE=test
BATCH=64
DEVICE=cuda
SAMPLES_ROOT=outputs/samples
RUN_ROOT=outputs/runs

NFES=(5 8 12 18 32 64)
SEEDS=(0 1 2 3 4)

cd "$(dirname "$0")/.."

n_total=$(( ${#NFES[@]} * ${#SEEDS[@]} ))
i=0
t0=$(date +%s)
for NFE in "${NFES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    i=$((i+1))
    elapsed=$(( $(date +%s) - t0 ))
    echo "[$(date +%H:%M:%S)] [$i/$n_total] ays_unipc nfe=$NFE seed=$SEED  elapsed=${elapsed}s"
    python -m autonomous_diffusion.experiments.run_generate \
      --contract "$CONTRACT" --dataset "$DATASET" --sampler ays_unipc \
      --nfe "$NFE" --seed "$SEED" --samples "$SAMPLES" --phase "$PHASE" \
      --samples-root "$SAMPLES_ROOT" --run-root "$RUN_ROOT" \
      --batch-size "$BATCH" --device "$DEVICE" >/dev/null
    python -m autonomous_diffusion.experiments.run_eval \
      --contract "$CONTRACT" --dataset "$DATASET" --phase "$PHASE" \
      --latest --retention seed0_only >/dev/null
  done
done
echo "[$(date +%H:%M:%S)] all $n_total runs complete"
