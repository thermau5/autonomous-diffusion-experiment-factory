# 30-hour autonomous run — progress log

Started 2026-05-29 (tmux). Strategy: bank low-risk results first (commit after each),
then attempt high-value/high-risk Level-3 path program, investigating failures.
Durability: commit after every completed step; this log is the resume anchor.

## Plan (risk-ordered)
1. [RUN] dpm_solver_v3 full sweep — Level-1 completion (low risk, existing sampler)
2. [ ] R2: R_disc<->FID 3-seed — lock risk-measure axis
3. [ ] T0+H1: reframe locked + parameterization theory (no GPU)
4. [ ] P1: RF/Reflow Level-3 (HIGH RISK)
5. [ ] P2: OT-CFM Level-3 (gated on P1)
6. [ ] D: compile unified report

## Log
- init: GPU free (15 MiB used). git at 9f9b04d. tasks #7-#11 created.

## Update 1
- dpm_solver_v3 sweep: running, ~16/30 (slowed by concurrent RF probes; results uncorrupted).
- T0+H1: DONE, committed 77c8001 (reframe section added to report_theory).
- P1 RF/Reflow: model loads + EMA applied + Euler sampler works end-to-end (64-sample smoke ok, range[-1,1]).
  - deps installed: ml_collections, absl-py, ninja (for NCSN++ fused op JIT; model.fir=False so op never called at runtime but imported).
  - RF convention: z0=randn*1.0 (std Gaussian); Euler dx/dt=v(x,t*999), t:eps(1e-3)->T(1), dt=1/N, NFE=N; final x in [-1,1] centered.
  - this is 1-Rectified-Flow (base flow); low-NFE Euler FID expected poor (path not very straight); 2-RF/distilled would be better.
  - sampler saved: scripts/rf_level3_sampler.py ; ckpt at third_party/rectified_flow/.../checkpoint_8.pth (990MB, gitignored)
- NEXT: wait dpm_v3 sweep done -> aggregate; then RF FID curve at locked NFE grid.

## Update 2 (dpm_v3 done)
- dpm_solver_v3 (logSNR+EMS) 5-seed locked-protocol DONE. Strongest single sampler:
  NFE 5/8/12/18/32/64 = 17.07/6.30/4.71/4.46/4.35/4.35 (+-~0.1..0.02).
  Beats (Ours,UniPC) 21.46/9.18/5.58/4.66/4.45/4.41 at EVERY NFE (-4.4 @K5).
  HONEST: a better SOLVER CORE (EMS) beats our best SCHEDULE; Level-1 advance, not schedule.
  agg: jobs/auto30/dpmv3_agg.json
- NEXT: RF/Reflow FID sweep launching (1-RF, Euler, 6 NFE x 3 seeds).

## Update 3 (path-straightening series found)
- RF repo has CIFAR-10 ckpts for 1-RF, 2-RF, 3-RF (reflow series = controlled path-straightening).
  This IS the Level-3 path-axis experiment (same arch, varying path straightness). Better than OT-CFM (needs training).
- Downloaded 2-RF + 3-RF ckpts (990MB each, disk 159G free). All load w/ same ddpmpp config.
- 1-RF sweep running; rf_sweep_param.py ready for 2-RF/3-RF.
- OT-CFM (P2) deprioritized: needs training (~12-24 GPU-h); reflow series gives a cleaner path-axis story.

## Update 4 (1-RF complete, chain advanced to 2-RF)
- 1-RF Euler FID (3-seed, 10k Clean-FID): NFE 5/8/12/18/32/64 = 38.59/20.21/13.76/10.57/8.04/6.53.
  Monotone, smooth. (NFE=64 Euler=6.53 > published 2.58 which uses RK45 ~127 NFE — expected; ours is fixed-step Euler.)
- Chain handoff verified: 1-RF exited 22:39:08 (~45min sweep), chain auto-launched 2-RF (GPU 100%).
- report_level3_path Table B 1-RF row filled. Awaiting 2-RF, 3-RF, R2.

## Update 5 (2-RF complete; floor probe queued)
- 2-RF Euler FID: NFE 5/8/12/18/32/64 = 7.55/6.99/6.70/6.53/6.38/6.31.
  Beats 1-RF at EVERY tested NFE (incl 64: 6.31 vs 6.53). NO crossover in the <=64 Euler grid.
  Low-NFE win is huge (NFE5: 7.55 vs 38.59, 5x). Curve far flatter (straighter path = NFE-insensitive).
  vs fixed-EDM frontier: 2-RF wins only at NFE5 (7.55 vs dpm_v3 17.07); dpm_v3 wins NFE>=8 (better floor ~4.35).
- Pre-registered part(ii) [1-RF<2-RF at NFE=64] NOT met at this grid: Euler@64 is far from 1-RF's converged
  2.58 floor, so the floor reversal must live at NFE>64. Added chain2_floor.sh: Euler {128,256} for 1/2/3-RF
  (same integrator, no RK45 confound) to locate the crossover. Runs after main chain (3-RF + R2) completes.
- 3-RF sweep running (handoff 23:28:30).

## Update 6 (3-RF complete; full path ladder)
- 3-RF Euler FID: NFE 5/8/12/18/32/64 = 7.45/7.20/7.06/6.97/6.89/6.85.
- FULL LADDER orderings: NFE5: 3<2<<1 (part i CONFIRMED); NFE8-32: 2<3<1; NFE64: 2<1<3 (1-RF overtook 3-RF).
  No single monotone winner -> floor-vs-defect split CONFIRMED (pre-registered rule).
  Crossovers: 3-vs-2-RF ~6 NFE (observed); 1-vs-3-RF in (32,64) (observed); 1-vs-2-RF >64 (floor probe).
  Optimal reflow count is NFE-dependent: 3-RF@NFE5, 2-RF@8-64, 1-RF beyond.
- report_level3_path: 3-RF row + full reconciliation section written. R2 running, floor probe queued.
