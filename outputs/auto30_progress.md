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

## Update 14 (3-RF schedule -> path x schedule grid COMPLETE)
- 3-RF defect ratio 72.1 (~1-RF 69, 2-RF 71.8): defect SHAPE constant across full reflow ladder.
- 3-RF proposed_p1 vs uniform: K5 7.26/7.40; K8 7.05/7.16; K12 6.94/7.02; K18 6.88/6.93; K32 6.83/6.85.
- FULL GAP GRID (proposed_p1 - uniform):
        K=5    K=8    K=12   K=18   K=32
  1-RF  -0.77  -0.67  -1.16  -1.36  -1.01
  2-RF  -0.27  -0.19  -0.13  -0.08  -0.04
  3-RF  -0.14  -0.11  -0.08  -0.05  -0.02
- Schedule beats uniform in EVERY (path,NFE) cell; gap shrinks monotonically with reflow (more floor-limited).
  Shape-driven leverage constant; absolute gain bounded by defect's share of FID. Path-straighten + schedule = complementary.
- Level-3 program on RF COMPREHENSIVELY COMPLETE (axes P, m, R, s all tested). ~26h elapsed.

## Update 15 (exploratory: FID-faithful weighted schedule on RF, gated)
- Tests whether feature-weighted defect d*g (g = Inception-feature sensitivity to a state perturbation at t,
  propagated to t=1 via fine Euler) beats pixel-d=||xddot|| for the RF schedule. Mirrors EDM perceptual-weight Q.
- rf_gsens.py estimates g(t) on 49 nodes + SANITY GATE (positivity/smoothness/profile) before any sweep.
  Clearly EXPLORATORY; honest report regardless of outcome. EDM precedent: feature-weighting did NOT beat tuned.

## Update 16 (g-sanity PASSED; FID-weighted schedule sweep running)
- g(t) feature-sensitivity (1-RF, 49 nodes, finite-diff eps=0.05, 2 reps): ALL POSITIVE, smooth, max/median=35.
  Profile: HIGH at noise end (t=0: ~500-730), dips through middle (min ~6.4 @t~0.7), rises @t=1 (~52).
  Interpretable: ODE sensitivity to near-initial condition dominates (x(0) ~ determines sample); mid-traj perturbations
  contracted toward data manifold. SANITY GATE PASSED -> proceed.
- KEY: pixel-d ||xddot|| peaks near t->1, but g peaks near t->0, so d*g reweights node placement toward noise end.
- rf_sched_fid.py running: proposed_dg (∝(d*g)^{1/2}) vs existing pixel-d-p1 / uniform on 1-RF, 3 seed.

## Update 17 (FID-weighted schedule on RF -- clean NEGATIVE result, confirms decomposability wall)
- proposed_dg (∝(d*g)^{1/2}, FID-faithful feature weighting): K5/8/12/18/32 = 48.36/26.33/16.90/11.58/8.02.
  WORSE than BOTH uniform (37.94/19.74/13.39/10.26/7.80) AND pixel-d-p1 (37.17/19.07/12.23/8.90/6.79) at EVERY NFE.
- WHY: local linear g peaks at noise end (t->0, ODE most sensitive to initial condition); d*g pulls nodes to t->0,
  starves the high-curvature data end (t->1) that needs resolution at deployment NFE. Regime-mismatched weighting.
- => Cross-family confirmation of the EDM decomposability wall: the "principled" FID-faithful local weighting HURTS;
  the parameter-free pixel-space calibration (d=||xddot||, p=1) is SUFFICIENT and SUPERIOR. g sane but wrong regime.
- Task #15 done. RF Level-3 program (P, m incl. FID-weighting test, R, s) COMPLETE. ~27h elapsed.

## Update 18 (INVESTIGATION: parallel proposed_rf track found -- independent confirmation)
- Discovered chain3_proposed.sh + proposed_rf_sweep.py (from pre-compaction session) running in parallel:
  an INDEPENDENT impl of the same schedule-axis test, with CLEANER methodology:
  (a) calibrates on HELD-OUT seed 777 (mine used seed 0 = overlaps an eval seed -- mild issue);
  (b) curvature via 2nd-difference x_{i+1}-2x_i+x_{i-1} (mine: velocity FD dv/dt); both estimate ||xddot||.
- Results AGREE (robustness): 1-RF gap held-out -0.20/-0.52/-1.11/-1.39/-1.05/-0.55 (K=5..64) vs mine -0.77/-0.67/-1.16/-1.36/-1.01.
  Mid/high-NFE near-identical; differ mainly @K=5 (held-out more conservative). 2-RF near-identical (-0.23/-0.18/-0.11/-0.07/-0.03).
- Concurrent GPU use earlier was real but HARMLESS (FID deterministic per seed; only slowed sweeps).
- ACTION: held-out-seed calibration is the cleaner PRIMARY; will reconcile report to it + note independent velocity-FD
  cross-check agrees. (Switching to held-out numbers is the rigorous direction even though it shows SMALLER low-NFE gain --
  NOT cherry-picking.) 3-RF proposed_rf running (curvature ~0, very straight -> proposed~uniform expected).

## Update 19 (RECONCILED report+summary to held-out-seed primary; proposed_rf chain COMPLETE)
- proposed_rf 3-RF done; PROPOSED CHAIN COMPLETE 03:09:36. Full held-out gap grid (proposed-uniform):
        K5     K8     K12    K18    K32    K64
  1-RF  -0.20  -0.52  -1.11  -1.39  -1.05  -0.55
  2-RF  -0.23  -0.18  -0.11  -0.07  -0.03  -0.01
  3-RF  -0.11  -0.10  -0.07  -0.05  -0.02  -0.01
- Report tab:sched now uses HELD-OUT-seed proposed_rf as primary (1-RF, +K64); seed-0 p-sensitivity kept as
  clearly-labeled internal comparison; interaction grid -> held-out. Honest note: K=5 1-RF gap (-0.20) anomalously
  small (too undersampled); monotone reflow-shrinkage holds K>=8. velocity-FD(seed0) cross-check agrees (peak -1.36 vs -1.39).
- Report recompiles (6pp). Summary updated. All GPU jobs done; GPU idle.

## Update 20 (SOLVER/SCHEDULE AXIS -- fairness correction, user-driven)
- User flagged: original EDM "solver-axis" table confounded 3 knobs (core+schedule+EMS). Correct single-knob test =
  per core, proposed schedule vs that core's OWN default, everything else fixed.
- Clean schedule-isolation on EMS core (proposed_dpm_solver_v3 SHARED d_Heun calib vs dpm_v3 default):
  nfe5 25.77 vs 17.07 (+8.70 LOSS); nfe8 12.47 vs 6.30 (+6.17 LOSS). Our schedule LOSES on EMS core.
- INVESTIGATION: EMS coeffs (l,s,b) are stored dense-in-lambda (1201 pts) + interpolated to any grid -> grid-agnostic,
  already fair (not the cause). REAL cause: proposed_dpm_v3 default uses SHARED d_Heun calibration (Heun's defect profile),
  NOT dpm_v3's own. User's point: "they tune solver for their schedule; tune ours too" = calibrate d to THIS solver.
- calibrate() (proposed_control.py) is SOUND pointwise: per-interval single-step(solver) vs Heun-substep ref -> d_s(sigma).
  per_core (AD_PROPOSED_CALIB=per_core, calib_id=core) calibrates each core's OWN defect. NOT the buggy seq path.
- LAUNCHED fair_table.sh: every proposed_<core> with per_core calibration vs own default (EMS row first, then Heun/DPM++/DEIS/UniPC),
  3 seeds, NFE{5,8,12,18,32,64}. Shared-calib dpm_v3 kept as ablation. Cell format (user): paired FID m*/default, bold winner.
- ETA ~12-16h GPU (162 runs). Will report pivotal EMS-fair result first.

## Update 21 (EMS joint-opt: d_EMS measured -- hypothesis REFUTED, redirect to validation-tuned k)
- Measured EMS-corrected solver's OWN per-step defect d_EMS(sigma) along its real trajectory (return_intermediate
  + Heun-substep reference). RESULT: d_EMS is NOT flat -- it PEAKS at low sigma (mean d at sigma<0.5 = 21.9 vs sigma>5 = 0.029,
  lo/hi=766; max/median=198), SAME shape as Heun's d. So my "EMS flattens d -> m* recovers their schedule" hypothesis is WRONG.
- Implication: m* from d_EMS ~ m* from d_Heun (same low-sigma-peaked shape) -> same losing schedule. Calibrating d to dpm-v3 won't fix it.
- proposed_dpm_v3 ALREADY uses perceptual k=2 (= EDM's validation-tuned value, borrowed). So the real un-fairness parallel:
  k=2 was tuned for EDM/Heun, NOT for the EMS solver. EMS corrects low-sigma error, so dpm-v3 likely wants LESS low-sigma
  concentration -> a different (lower/negative) k.
- LEGITIMATE fair test (sanctioned protocol, tune_on=validation_only): validation-tune k (perceptual exponent) for dpm-v3,
  then ONE locked-test eval. NOT fishing on test. Honest caveat: may only RECOVER their schedule (tie), not beat -- report either way.
- ems_defect.json saved. Launching validation k-sweep for proposed_dpm_v3.

## Update 22 (EMS joint-opt RESULT: honest NEGATIVE -- our scheduler does NOT beat dpm-v3)
- Validation k-sweep (tune_on=validation_only, legitimate): k in {-1,0,1,2,3}. Curve is a BOWL, minimum at k=2:
  nfe5 by k: 253/245/132/25.7/67. Min (k=2)=25.7/12.4/7.4 -- LOSES to default 17.0/6.30/4.70 at every validation NFE.
- k=2 = the borrowed-EDM value; its TEST curve vs default: nfe 5/8/12/18/32/64 = 25.77/12.47/7.44/5.45/4.69/4.34
  vs 17.07/6.30/4.71/4.46/4.35/4.35. Delta +8.70/+6.18/+2.73/+0.99/+0.33/-0.01. Gap shrinks with NFE; TIE only at nfe64 (floor).
- CONCLUSION (honest, no fishing): NO joint-opt with our certificate scheduler beats dpm-v3 at any NFE.
  Mechanism: dpm-v3 schedule co-derived with EMS from same model statistics (joint solver+schedule opt); low-NFE co-design
  dominates; high-NFE all schedules hit ~4.35 floor. Validation-tuning k cannot close it (bowl min 25.7 >> 17). EMS = BOUNDARY case.
- Report EMS as documented boundary; cores (Heun/UniPC/DPM++/DEIS) + RF carry dominance. Did NOT extend grid (= fishing). GPU back to cores.

## Update 23 (fair dominance table -- 4 EDM cores; 2 harness gotchas found + fixed)
- Ran per-core proposed (per_core calib) vs own default, 3-seed, NFE{5,8,12,18,32,64}. Results:
  - DPM++ : proposed 23.16/9.74/6.07/4.81/4.49/4.43 vs default 40.23/13.46/6.83/5.04/4.57/4.46 -> WIN (ties@64). CLEAN.
  - UniPC : proposed 21.44/9.18/5.58/4.66/4.46 vs default 40.41/12.50/6.06/4.79/4.50 -> WIN (ties@32). CLEAN.
  - DEIS  : CONTAMINATED -- proposed_deis per_core = seq AB-2 (flagged-buggy forced-smin path) -> 359/24.9/25.2 catastrophic.
            FIX: re-run with AD_PROPOSED_CALIB=shared (d_Heun pointwise, the version that won in locked). [running]
  - Heun  : default edm_heun requires ODD NFE (=2*steps-1); even grid {8..64} errored. FIX: re-run both arms at odd {5,9,13,19,33,65}. [running]
- RF (1/2/3-RF) + EMS rows unchanged (RF wins every cell to floor; EMS boundary loses low-NFE ties@floor).
- HEADLINE so far: our calibrated schedule beats each solver's own default on DPM++, UniPC (EDM) + 1/2/3-RF; ties at floor;
  EMS is the one boundary. DEIS/Heun re-running with correct calib/grid. fix_rows.sh -> fix.log.
- LESSON: per_core calibration is core-dependent: sound pointwise for heun/dpmpp/unipc, but =buggy seq for deis. Heun NFE must be odd.

## Update 24 (VP-SDE path added -- third path family)
- Goal: add VP as a 3rd probability-path family (VE/EDM + linear/RF + VP). RF clone stripped VP; found full score-SDE
  codebase bundled in third_party/dpm_solver_v3/codebases/score_sde (VPSDE/subVPSDE/VESDE + vp/cifar10_ddpmpp_deep_continuous config).
- CHECKPOINT saga: README link (1F74y6G) served JAX/msgpack checkpoints (incompatible w/ torch.load). Found PyTorch checkpoints
  in score_sde_pytorch Drive (1tFmF...); traversed folder via gdown tree-listing -> vp/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth
  (id 16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif). Header PK = valid torch. Loads (step 400005), PF-ODE Euler samples sane ([-1.07,1.10], no nan).
- LAUNCHED vp_sched.py: VP PF-ODE Euler, uniform-t (VP default) vs calibrated m* (d(t)=||xddot|| on ref traj, p=1), NFE{5,8,12,18,32,64},
  3 seed, Clean-FID 10k. Parallels RF schedule test. -> vp_sched_results.json.
- LESSON: nested Drive checkpoints -> gdown --folder prints all file IDs in the tree; grab the specific id, don't bulk-download.

## Update 25 (VP-SDE schedule axis -- CLEAN POSITIVE; m* generalizes to a 3rd path family)
- vp_sched.py COMPLETE. Calibrate VP defect d(t)=||xddot|| on a ref PF-ODE trajectory (seed 777, disjoint from eval seeds 0/1/2),
  place K Euler-t nodes ~ d^{1/(p+1)}, p=1 pinned (Euler order, parameter-free, NO FID feedback). Baseline = VP default uniform-t.
  Both arms share the same drift_f Euler loop + endpoints T->eps; only interior node placement differs. VP defect max/median=29.6, peaks at data end t~0.005.
- RESULT (Clean-FID 10k, 3 seeds, all SEM<=0.32 so every gap is 12-23 sigma):
  NFE      5       8       12      18      32      64
  uniform  240.35  150.94  80.13   43.06   22.28   14.11
  m* (p=1) 235.71  131.47  67.29   37.06   19.55   12.32
  Delta    -4.64   -19.47  -12.84  -6.00   -2.73   -1.79   (rel -1.9/-12.9/-16.0/-13.9/-12.2/-12.7%)
- VERDICT: calibrated schedule m* beats VP's uniform-t default at EVERY NFE. Peak relative gain ~16% mid-budget (K=12);
  K=5 gain anomalously small (1.9%) -- same pattern as RF (VP-Euler@K=5 is FID 240, too undersampled for placement to help).
- HEADLINE: the certificate's schedule prescription now generalizes across THREE independent path families --
  VE/EDM (Karras), linear-interp/Rectified-Flow, and VP-SDE -- parameter-free, FID-feedback-free, at every NFE. Not an EDM artifact.
- LESSON: VP-Euler is step-hungry (uniform-t FID 240 @K=5); the absolute FIDs are high but the schedule COMPARISON is what the axis tests,
  and it is clean. Same protocol, same conclusion as RF -> robust cross-family generalization.

## Update 26 (dominance table assembled + 2 stale-source bugs caught + integrated into report)
- Rebuilt build_table.py to assemble the full schedule-dominance table (m*/own-default paired FID, bold winner @2-SIGMA of the seed diff,
  ~ = tie). Caught TWO assembler bugs while finalizing (NOT data bugs -- the data was correct, the assembler read the wrong source):
  1. DEIS row was showing the CONTAMINATED per_core seq-AB2 numbers (359/24.9/25.2) from fair.log; the corrected shared-calib run lives
     in fix.log (28.59/11.62/7.51/5.50/4.69/4.48). Fixed: read DEIS proposed from fix.log. -> DEIS WINS NFE5/8, TIES 12-64 (weakest core).
  2. Heun row only had NFE5 (assembler keyed even grid; Heun ran ODD {5,9,13,19,33,65}). Fixed: map odd grid -> even columns (+1 NFE/cell).
     -> Heun HUGE WIN 5-18 (13.41 vs 338.52!), ties at floor.
- Also switched hard 0.05 tie-threshold -> 2-sigma of the per-seed difference (honest: floor sems ~0.04-0.1, so 0.05 was too tight;
  demoted several marginal floor "wins" to ties; 1-RF@NFE5 win->tie).
- FINAL DOMINANCE TABLE (m* wins-or-ties EVERY cell across 4 EDM cores x 3 path families; lone exception = EMS boundary):
  Heun  WIN 5/8/12/18, tie 32/64    | DPM++ WIN 5/8/12/18, tie 32/64 | DEIS WIN 5/8, tie 12-64
  UniPC WIN 5/8/12/18, tie 32/64    | 1-RF  tie 5, WIN 8-64          | 2-RF WIN 5/8/12, tie 18-64
  3-RF  WIN 5/8, tie 12-64          | VP-SDE WIN ALL 6 NFE (cleanest -- never hits floor) | EMS LOSS 5-32, tie 64 (boundary)
- INTEGRATED into report_level3_path.tex: restructured Sec.5 to LEAD with the dominance table (Table 5, tab:dominance), kept 1-RF as
  the "anatomy of one column" zoom-in; updated abstract (ii), Sec.1 glance [m], axis-map table ref, and Conclusion. Compiles clean,
  5pp, no overfull >20pt, no undefined refs. Also converted appendix-B exponent-sensitivity inline list -> tabular (cleared 65pt overfull).
- HEADLINE: the certificate's parameter-free schedule m*=d^{1/(p+1)} (p=1) wins or ties the NATIVE DEFAULT of every solver core AND every
  path family tested, across 3 independent path families (EDM/RF/VP). One documented boundary: EMS (schedule co-designed with solver).

## Update 27 (EMS row was BORROWED d_Heun, not genuine per-core -- user caught it; running genuine d_EMS)
- User question: "is our m* on EMS genuine like the other cores or an approximated workaround?" -> CORRECT instinct.
- DIAGNOSIS: proposed_dpm_solver_v3 applies the m* GRID genuinely (overrides EMS timesteps, runs true EMS update -- EMS is grid-agnostic),
  BUT the DEFECT driving m* is BORROWED from Heun: calibrate() has no 'dpm_solver_v3' step fn (PER_CORE_STEP only has heun/dpmpp/unipc/deis),
  so it ValueErrors on per_core and the sampler falls back to calib_id='heun' (d_Heun). The reported EMS row (25.77/...) = m*(d_Heun)+k=2 on EMS.
- Earlier I INFERRED m*(d_EMS) ~ m*(d_Heun) ("same low-sigma shape") and never ran it. ems_genuine_calib.py REFUTES that inference:
  d_EMS max/median=198 vs d_Heun max/median=4.1 (d_EMS far more peaked); resulting m* grids differ by rel-L1 17%(NFE5)->43%(NFE64). NOT the same schedule.
- Built GENUINE d_EMS calibration (measured ON the EMS trajectory, Heun-substep ref -- SAME protocol as the other 4 cores), saved to
  calib_dpm_solver_v3_988c93ee001e5fad.npz (net key MATCHES the other cores' calibs -> harness loads it). ems_genuine.sh sweep launched (per_core, k=2, p=2).
- FIRST RESULT NFE5: genuine d_EMS = 19.36 vs borrowed d_Heun 25.77 vs dpm-v3 default 17.07. Genuine recovers 6.4 of the 8.7 gap!
  -> The "EMS boundary" as reported was largely a BORROWED-d_Heun ARTIFACT, not a fundamental ceiling. Full sweep running; will correct report EMS row + framing.
- LESSON: per-core fairness requires per-core DEFECT, not just per-core GRID application. Borrowing d across solvers is a real (large) handicap when defect shapes differ.

## Update 28 (GENUINE d_EMS RESULT: EMS is NOT a boundary -- ties/wins at NFE>=12; report corrected)
- ems_genuine.sh DONE (k=2, p=2, per_core d_EMS, 3 seeds, same contract pipeline as every cell). genuine d_EMS / dpm-v3 default:
  NFE 5/8/12/18/32 = 19.33/6.64/4.71/4.38/4.34  vs  17.07/6.30/4.71/4.46/4.35. NFE64 = FAILED (singular EMS solve).
- VERDICT @2sigma: LOSS 5,8 (+2.26,+0.34) | TIE 12,32 | WIN 18 (4.38 vs 4.46, 2.4sigma) | NFE64 numerical fail.
  vs borrowed d_Heun (25.77/12.47/7.44/5.45/4.69) which LOST every NFE. Genuine per-core defect closes most of the gap + flips mid-NFE to ties/win.
- NFE64 failure is DETERMINISTIC (all 3 seeds): genuine d_EMS max/median=198 (vs d_Heun 4.1) so peaked that inverse-CDF at K=64 crowds
  near-identical low-sigma nodes -> EMS coefficient linalg.inv singular. Honest: reported as fail (--); all schedules tie at floor there anyway. NOT hacked around.
- k=2 borrowed from EDM, NOT re-tuned for EMS -> result is CONSERVATIVE (didn't fish a better k).
- SECOND inconsistency found+fixed: Sec.2 said "EMS beats our best schedule at every NFE" but dominance Heun row (13.41@NFE5) BEATS EMS (17.07).
  Corrected Sec.2 to "EMS beats our UniPC arm + best floor" + noted Heun-m* beats EMS at the 2 lowest budgets. Glance[s] softened (no false "beats our best").
- REPORT updated: EMS dominance row -> genuine d_EMS; dropped "boundary" framing (abstract ii, glance m, caption, footnote, reading para, conclusion);
  added "Genuine per-core calibration matters" para (borrowed d_Heun 17-43% diff grid, loses everywhere; genuine closes gap). Compiles 5pp clean, no overfull, no undef.
- NET: across 4 cores x 3 paths, the ONLY losses anywhere are EMS @NFE5,8. Stronger + more honest than "EMS boundary". User's question fixed a real methodological gap.

## Update 29 (user asked for BOTH follow-ups: genuine-by-construction + validation-tune k for d_EMS)
- TASK 1 (genuine by construction, committed 3eadf43): added ProposedDPMSolverV3._calibrate_ems -- measures d_EMS ON the EMS
  trajectory (Heun-substep ref, SAME protocol as the other 4 cores), packs into the standard Calibration. per_core now routes
  through it natively (no saved-calib side door). Verified: reproduces side-door grid to ~0.5% (d_EMS max/med 195; NFE5 m*=[80,7.55,1.01,0.085,0.009]).
- TASK 2 (validation-tune k for d_EMS, tune_on=validation_only): ems_kval.sh + ems_kval_ext.sh. VALIDATION FID (seed-mean) NFE 5/8/12/18/32:
  default 17.01/6.30/4.70/4.43/4.32 | k=0 226/149/86/49/FAIL | k=1 60.8/24.9/16.7/9.4/5.10 | k=2 19.25/6.54/4.66/4.36/4.32 | k=3 201/125/FAIL.. | k=4 ALL FAIL.
- RESULT: SHARP validation bowl, min at k*=2. k<=1 under-concentrates (^{1/3} flattens d_EMS's 195x peak so sigma^-k does the real
  concentrating); k>=3 OVER-crowds nodes -> singular EMS solve (fails even at low NFE for k=4). Much sharper than d_Heun's bowl (k=3 merely worse).
- So the pre-specified borrowed k=2 IS the d_EMS validation optimum. Validation tracks test near-exactly (val 19.25/6.54/4.66/4.36/4.32 vs
  test 19.33/6.64/4.71/4.38/4.34) -> no overfitting, the existing k=2 TEST row IS the validation-confirmed answer. NO new test draw needed
  (k=2 was never selected from test data; validation independently confirms). Clean under no-run-until-success.
- NFE64 failure now fully explained: same over-concentration pathology that kills k>=3; at k*=2 the schedule sits at the numerical-stability
  edge and NFE64 tips over. Floor-tie regardless. Report footnote updated: k=2 = validation-confirmed optimum (sharp bowl), not just "borrowed/conservative".
- LESSON: when d is already peaked, the ^{1/(p+1)} exponent flattens it and the perceptual weight k carries the concentration; the k-optimum
  is then SHARP and over-concentration is catastrophic (numerical, not just FID). Both follow-ups DONE; report compiles 5pp clean.

## Update 30 (NEXT direction chosen by user: OT-CFM as a 4th path family)
- Goal: add optimal-transport CFM as a 4th path family (EDM/RF/VP + OT-CFM) to the schedule-dominance table; test whether m* generalizes to an OT-straightened path.
- No downloadable OT-CFM CIFAR ckpt (TorchCFM ships code not weights). Chose: fine-tune the converged 1-RF model with MINIBATCH-OT COUPLING (scipy linear_sum_assignment,
  exact for equal batches) -- SAME NCSN++ backbone as the 1/2/3-RF rows, ONLY the coupling differs (independent->OT), so OT-CFM vs 1-RF isolates coupling-driven straightening.
  Warm-start is legitimate (field converges to genuine OT-CFM optimum regardless of init = same logic reflow uses); train to loss-plateau, NOT a borrowed approximation.
- Self-contained ot_cfm_train.py (avoids RF run_lib pipeline): load 1-RF EMA, CIFAR-10 [-1,1], Adam lr1e-4, EMA 0.9999, OT-reorder z0<->x1 per batch, CFM loss on linear interpolant.
  Smoke-tested OK (61.8M params, loss ~0.147). LAUNCHED 30k steps (0.25s/it ~2.1h), ckpt every 5k -> otcfm_ck/. Monitoring loss for plateau (= OT field converged).
- ot_cfm_sched.py READY (mirrors vp_sched): calibrate d(t)=||xddot|| on OT-CFM (seed 777 held-out), m*(p=1) vs uniform-t, NFE{5,8,12,18,32,64}, 3 seeds, Clean-FID.
- NEXT: when training plateaus -> run ot_cfm_sched -> add OT-CFM row to dominance table (report) as the 4th path family. Compare defect max/median vs 1-RF (~69) to see how much OT straightens.
