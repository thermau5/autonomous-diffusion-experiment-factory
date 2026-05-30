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
