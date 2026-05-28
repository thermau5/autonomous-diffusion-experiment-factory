#!/usr/bin/env bash
# Round 6: confirm the finite-N calibrated certificate (PDF v4.1) beats the
# shared-d_Heun heuristic for DEIS at matched NFE. proposed_deis_seq at the
# locked NFE grid x 5 seeds x 10k samples. Grids are precomputed/cached.
set -euo pipefail
CONTRACT=contracts/benchmark_contract.yaml
cd "$(dirname "$0")/.."
NFES=(5 8 12 18 32 64)
SEEDS=(0 1 2 3 4)
i=0; n=$(( ${#NFES[@]} * ${#SEEDS[@]} )); t0=$(date +%s)
for NFE in "${NFES[@]}"; do for SEED in "${SEEDS[@]}"; do
  i=$((i+1)); el=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] [$i/$n] proposed_deis_seq nfe=$NFE seed=$SEED elapsed=${el}s"
  python -m autonomous_diffusion.experiments.run_generate \
    --contract "$CONTRACT" --dataset cifar10 --sampler proposed_deis_seq \
    --nfe "$NFE" --seed "$SEED" --samples 10000 --phase test \
    --batch-size 64 --device cuda >/dev/null
  python -m autonomous_diffusion.experiments.run_eval \
    --contract "$CONTRACT" --dataset cifar10 --phase test --latest \
    --retention seed0_only >/dev/null
done; done
echo "[$(date +%H:%M:%S)] all $n runs complete"
