#!/usr/bin/env bash
# Round 6 fill #3: calibrated-sequence (Theorem B) on DPM-Solver++ and UniPC
# cores, to quantify the A approximately B gap vs the locked pointwise m_s*.
set -euo pipefail
CONTRACT=contracts/benchmark_contract.yaml
cd "$(dirname "$0")/.."
NFES=(5 8 12 18 32 64); SEEDS=(0 1 2 3 4)
SAMPLERS=(proposed_dpmpp_seq proposed_unipc_seq)
i=0; n=$(( ${#SAMPLERS[@]} * ${#NFES[@]} * ${#SEEDS[@]} )); t0=$(date +%s)
for S in "${SAMPLERS[@]}"; do for NFE in "${NFES[@]}"; do for SEED in "${SEEDS[@]}"; do
  i=$((i+1)); el=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] [$i/$n] $S nfe=$NFE seed=$SEED elapsed=${el}s"
  python -m autonomous_diffusion.experiments.run_generate \
    --contract "$CONTRACT" --dataset cifar10 --sampler "$S" \
    --nfe "$NFE" --seed "$SEED" --samples 10000 --phase test \
    --batch-size 64 --device cuda >/dev/null
  python -m autonomous_diffusion.experiments.run_eval \
    --contract "$CONTRACT" --dataset cifar10 --phase test --latest \
    --retention seed0_only >/dev/null
done; done; done
echo "[$(date +%H:%M:%S)] all $n runs complete"
