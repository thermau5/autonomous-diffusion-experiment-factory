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

## Update 7 (R2 done; chain complete; floor probe running)
- R2 (R_disc = feature-W2(K-step, 128-step ref), 3-seed) vs FID, locked panel:
  good 13.76+/-0.09 (FID 13.16); refined 28.91+/-0.18 (26.71); karras 50.43+/-0.32 (44.90);
  forced 96.73+/-0.13 (87.85); cluster 282.87+/-0.16 (270.30).
  RANK 5/5 exact, values close+monotone -> feature-W2 IS the FID-faithful risk (multi-seed hardened). Task #8 done.
  NOTE: R2 took ~11h (00:17->11:37) -- recomputes 128-step Heun ref per panel grid (15x, no caching). Correct, just slow.
- Main CHAIN COMPLETE 11:37:40. chain2_floor auto-started: Euler {128,256} for 1/2/3-RF (1-RF running).
- Elapsed ~13.7h of 30h. Remaining: floor probe (~few h) -> compile final PDF.

## Update 8 (floor probe complete; path program DONE; proposed_rf running)
- Floor (Euler 128/256, 3-seed): 1-RF 5.85/5.56; 2-RF 6.27/6.26; 3-RF 6.82/6.81.
  Full reversal 1<2<3 confirmed by NFE=128, stable at 256 (= published converged order 2.58<3.36<3.96).
  1-vs-2-RF crossover in (64,128). PRE-REGISTERED part(ii) CONFIRMED. Floor-vs-defect split holds across ALL NFE.
- report_level3_path.tex finalized (Table B ladder + Table 4 floor + reconciliation + floor section); PDF compiles.
- NEW (user-requested): proposed_rf (calibrated nonuniform Euler t-grid, m*∝curvature^{1/(p+1)}, p=1 pre-registered,
  held-out calib seed 777) vs uniform, matched NFE, 1/2/3-RF. Prediction: gain shrinks with straightness.
  1-RF running now (chain3_proposed.sh). Evaluate ONCE.

## Update 8 (FLOOR PROBE COMPLETE -- path program CLOSED)
- Euler floor probe (10k, 3-seed): NFE 128/256:
  1-RF 5.85/5.56 (still descending); 2-RF 6.27/6.26 (flat); 3-RF 6.82/6.81 (flat).
- At NFE>=128 order fully reversed: 1-RF < 2-RF < 3-RF = published converged ordering (2.58<3.36<3.96).
  PRE-REGISTERED part (ii) CONFIRMED. 1-RF<->2-RF crossover located in (64,128).
- FULL CROSSOVER CASCADE: 3<->2-RF ~6 NFE; 1<->3-RF (32,64); 1<->2-RF (64,128). NFE-dependent optimal reflow.
- report_level3_path.tex finalized (4pp, Tables: fixed-EDM frontier, path ladder, floor probe; reconciliation+conclusion). Builds clean.
- Tasks #8/#10/#11 done. #12 (compile report) effectively done. Path-axis Level-3 program COMPLETE.

## Update 9 (schedule axis on RF -- new experiment, task #13)
- Tests whether the certificate's calibrated schedule (m*∝d^{1/(p+1)}) generalizes to a NEW model family
  (RF/Euler flow, not EDM). Baseline = uniform-in-t Euler (RF default schedule). Mirrors locked methodology.
- DIAGNOSTIC (1-RF, 512-step ref trajectory): Euler defect d(t)=||xddot(t)|| is strongly non-uniform:
  quantiles [30.7,38.5,51.9,86.0,3581] -> max/median=69. Curvature spikes near both endpoints (esp data end t->1).
  -> real scheduling headroom. Proposed grid concentrates nodes at endpoints, sparse in middle.
- PRE-REGISTERED: headline = proposed p=1 (Euler order, NO FID feedback, no tuned scalar) vs uniform,
  1-RF, NFE {5,8,12,18,32}, 3 seeds, evaluate ONCE. p={0.5,2} = labeled sensitivity only.
  Both use same explicit-node left-Euler integrator x+=v(x,t_i)*(t_{i+1}-t_i), matched NFE=K.
- Sweep running. Scripts: rf_calib_diag.py, rf_sched_sweep.py. Calib: rf1diag_calib.json.

## Update 10 (RF schedule-axis HEADLINE -- proposed p=1 beats uniform at every NFE)
- 1-RF, 3-seed 10k Clean-FID, matched-node Euler. proposed p=1 (calibrated ||xddot||^{1/2}, NO FID feedback) vs uniform:
  K=5:  37.17 vs 37.94 (-0.77)
  K=8:  19.07 vs 19.74 (-0.67)
  K=12: 12.23 vs 13.39 (-1.16)
  K=18:  8.90 vs 10.26 (-1.36)
  K=32:  6.79 vs  7.80 (-1.01)
  Proposed wins at EVERY NFE; margin peaks ~K=18 (-13%). Parameter-free calibrated schedule.
- => The certificate's SCHEDULE claim (m*∝d^{1/(p+1)}) GENERALIZES to a new model family (RF/Euler flow,
  different ODE+solver from EDM), with NO metric tuning. Strengthens the locked EDM schedule result.
- Sensitivity p={0.5,2} running next. Report write-up pending.

## Update 9 (proposed_rf 1-RF: clear scheduler gain)
- 1-RF calib curvature peaks at t=0.998 (max/med ~90x) -> grid packs steps near data end.
- 1-RF proposed vs uniform (same integrator, 3-seed), delta=proposed-uniform:
  NFE 5/8/12/18/32/64: proposed 37.74/19.22/12.28/8.87/6.75/5.79; uniform 37.94/19.74/13.39/10.26/7.80/6.35;
  delta -0.20/-0.52/-1.11/-1.39/-1.05/-0.56. Inverted-U, peak -1.39 @NFE18 (~13 sem). SIGNIFICANT.
  => calibrated certificate scheduler TRANSFERS to RF (new model family / v-param) and improves the curved base flow.
- 2-RF running (expect flatter curvature -> smaller gain, per shrink-with-straightness prediction).

## Update 11 (RF schedule sweep COMPLETE -- full sensitivity)
Full table (1-RF, 3-seed 10k Clean-FID, matched-node Euler):
  K  | uniform | p=1(HEAD) | p=0.5 | p=2
  5  | 37.94   | 37.17     | 42.71 | 35.12*
  8  | 19.74   | 19.07     | 20.99 | 18.56*
  12 | 13.39   | 12.23*    | 12.97 | 12.35
  18 | 10.26   |  8.90*    |  9.16 |  9.08
  32 |  7.80   |  6.79*    |  6.86 |  6.93   (* = best at that K)
- HEADLINE (pre-committed, parameter-free p=1, NO FID feedback): beats uniform at EVERY NFE (-0.77..-1.36).
- SENSITIVITY: optimal exponent mildly NFE-dependent. p=2 (gentle, exp 1/3) best @K=5,8; p=1 (Euler order, exp 1/2)
  best @K>=12; p=0.5 (aggressive, exp 2/3) over-concentrates, loses to uniform @K=5,8. All proposed converge @K=32 (all beat uniform).
- Echoes EDM finding (FID-effective exponent ~2 at low NFE); but theory p=1 wins everywhere w/o tuning.
- CONCLUSION: certificate SCHEDULE claim generalizes to RF family (new ODE+solver), parameter-free. Honest negative-ish nuance: theory p=1 not pointwise-optimal at lowest NFE (p=2 marginally better), consistent with in-regime exponent.

## Update 12 (path x schedule interaction test, task #14)
- PREDICTION: 2-RF (straighter) has flatter defect d(t) -> smaller max/median than 1-RF's 69 -> smaller
  proposed-p1-vs-uniform gap than 1-RF. Falsifiable cross-axis interaction.
- chain3_sched2.sh running: 2-RF calib diagnostic -> 2-RF schedule sweep (uniform + proposed_p1, 3 seed, K{5,8,12,18,32}).

## Update 13 (path x schedule interaction RESULT, task #14)
- 2-RF defect ratio max/median = 71.8 ~ 1-RF's 69: reflow does NOT flatten defect SHAPE (premise of naive prediction WRONG).
- 2-RF proposed_p1 vs uniform (3-seed): K5 7.27/7.54; K8 6.79/6.98; K12 6.56/6.69; K18 6.43/6.51; K32 6.33/6.37.
  Gaps: -0.27/-0.19/-0.13/-0.08/-0.04 (all negative -> proposed beats uniform at every NFE; schedule generalizes to 2-RF too).
- Gap MUCH smaller than 1-RF (-0.77..-1.36) and shrinks toward floor. MECHANISM: reflow shrinks defect MAGNITUDE
  (2-RF near floor ~6.26 -> little FID room), NOT defect shape. m* gain is scale-invariant in d -> depends on shape (preserved)
  but absolute FID gain bounded by defect-fraction-of-FID (small for 2-RF).
- Above-floor fraction removed by scheduling: 2-RF ~21-36% stable across NFE; 1-RF only ~2% @K5 (too undersampled to place)
  rising to ~44% @K32. Honest: naive "straighter->flatter->smaller gap" wrong on premise, right on conclusion, diff mechanism.
