# Autonomous Diffusion — experiment factory

Autonomous benchmark design and **locked real-image verification** for a
Pareto-optimal certified generative-transport scheduler. The agent runs as

```
Theory  ->  Benchmark Design  ->  Implementation  ->  Validation Search
        ->  Locked Real-Image Test  ->  Paper Table/Figure
```

and **not** as "run until success." Validation tuning is unlimited; the locked
test is evaluated exactly once per declared seed. See
[`METHODOLOGY.md`](METHODOLOGY.md) and the runtime guards in
`src/autonomous_diffusion/critic/guards.py`.

## Theory

Generative dynamics
```
dx_t / dt = f_theta(x_t, t, u_t),
```
discretised as `x_{i+1} = Phi^{u_i}_{Δt_i}(x_i; theta)`. The proposed scheduler
solves
```
u*  ∈  argmin_u  C(u)   s.t.   R̂_val(u) + B_n(u, δ) ≤ ε
```
with the Pareto form
```
u*_λ ∈ argmin_u  D̂(P_u, P_data) + λ·C(u) + γ·[ R̂_val(u) + B_n(u, δ) - ε ]_+ .
```
See `docs/Autonomous Diffusion - Autonomous Diffusion.pdf` (KL/NLL variant) and
`docs/Autonomous Diffusion - Reformulating Diffusion as Control.pdf`
(Wasserstein variant) for the full theory.

## Benchmarks (locked by `contracts/benchmark_contract.yaml`)

- **Datasets:** CIFAR-10 32², FFHQ 64², AFHQv2 64² — frozen pretrained EDM nets.
- **Baselines:** EDM-Euler, EDM-Heun, DDIM, DDPM, DPM-Solver, DPM-Solver++,
  UniPC, DEIS, PNDM, Restart, plus uniform / Karras schedule controls.
- **Primary metric:** Pareto-AUC over FID vs NFE (Clean-FID). Secondary:
  Clean-FID at matched NFE. Always reported as mean ± SEM over the declared
  seed grid `[0, 1, 2, 3, 4]`.

## Quick start

```bash
make setup                          # conda env + clone NVlabs/edm + pip install -e .
make test                           # unit tests
make smoke DATASET=cifar10 SAMPLES=1024 SAMPLER=edm_heun NFE=35
make validation_sweep DATASET=cifar10 HOURS=10
make select_validated_config        # writes outputs/freeze_record.yaml
make freeze_contract
make locked_test DATASET=cifar10    # refuses if freeze_record drifts
make report
```

Or end-to-end:

```bash
make autonomous_real_image_run CONTRACT=contracts/benchmark_contract.yaml HOURS=12
```

## Layout

```
contracts/      benchmark_contract.yaml          (locked test spec)
configs/        per-dataset, per-phase overrides
src/autonomous_diffusion/
  critic/       guards.py freeze.py              (no-run-until-success enforcement)
  samplers/     base.py edm.py proposed_control.py ddim.py dpm_solver.py ...
  models/       edm_loader.py
  metrics/      clean_fid.py pareto.py risk.py
  experiments/  run_generate.py run_eval.py run_sweep.py run_locked_test.py
  report/       make_tables.py make_figures.py write_latex.py
tests/          test_*.py
outputs/        runs/ samples/ metrics/ figures/ paper/    (gitignored)
third_party/    edm/ (NVlabs source for unpickling pretrained nets)  (gitignored)
docs/           Autonomous Diffusion PDFs
```
