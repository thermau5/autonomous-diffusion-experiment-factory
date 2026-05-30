#!/usr/bin/env bash
# Level-1 completion: dpm_solver_v3 (logSNR+EMS) full locked-protocol sweep.
set -euo pipefail
CONTRACT=contracts/benchmark_contract.yaml
cd "$(dirname "$0")/.."
NFES=(5 8 12 18 32 64); SEEDS=(0 1 2 3 4)
i=0; n=$(( ${#NFES[@]} * ${#SEEDS[@]} )); t0=$(date +%s)
for NFE in "${NFES[@]}"; do for SEED in "${SEEDS[@]}"; do
  i=$((i+1)); el=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] [$i/$n] dpm_solver_v3 nfe=$NFE seed=$SEED elapsed=${el}s"
  python -m autonomous_diffusion.experiments.run_generate \
    --contract "$CONTRACT" --dataset cifar10 --sampler dpm_solver_v3 \
    --nfe "$NFE" --seed "$SEED" --samples 10000 --phase test \
    --batch-size 64 --device cuda >/dev/null 2>>outputs/dpmv3_sweep.err
  python -m autonomous_diffusion.experiments.run_eval \
    --contract "$CONTRACT" --dataset cifar10 --phase test --latest \
    --retention seed0_only >/dev/null 2>>outputs/dpmv3_sweep.err
done; done
echo "[$(date +%H:%M:%S)] all $n runs complete"
