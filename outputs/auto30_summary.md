# 30-Hour Autonomous Run — Executive Summary

**Window:** 2026-05-29 ~21:54 → 2026-05-30 (single RTX 4090, GPU-serialized).
**Theme:** test the generic excess-risk certificate on *every axis it can be tested on*, using a new model family (Rectified Flow) to complement the locked EDM benchmark. Detailed log: `outputs/auto30_progress.md`. Full write-up: `outputs/paper/new_generic/report_level3_path.tex` (5 pp). All steps committed + pushed.

The certificate: `Q = Q₀ + (c_stat/n)∫a/ρ + c_disc∫d/mᵖ`, optimum `m*∝d^{1/(p+1)}`, hierarchy `h→a, P→(a,d,Q₀), s→d, R→(a,d)`. We tested four axes.

---

## 1. Solver axis (s) — fixed-EDM frontier
**dpm_solver_v3 (logSNR + EMS), 5-seed locked protocol:** NFE 5/8/12/18/32/64 = **17.07/6.30/4.71/4.46/4.35/4.35**, beating our best schedule (Ours,UniPC) at *every* NFE. Honest: a better **solver core** (EMS) beats our best **schedule** on the fixed EDM path — a Level-1 advance, consistent with the certificate (s enters d). Sets the bar for the path axis.

## 2. Risk axis (R) — R_disc ↔ FID, multi-seed hardened
**R2 (3-seed):** the directly-measurable Wasserstein risk `R_disc = feature-W₂(K-step, 128-step ref)` tracks FID **5/5 in rank** and closely in value across a 5-grid panel (good 13.76↔13.16, …, cluster 282.9↔270.3). → feature-W₂ **is** the FID-faithful risk. Risk-measure axis settled.

## 3. Path axis (P) — reflow series 1/2/3-RF = controlled path-straightening
Full Euler FID ladder (10k Clean-FID, 3 seeds):

| NFE | 5 | 8 | 12 | 18 | 32 | 64 | 128 | 256 |
|-----|----|----|----|----|----|----|-----|-----|
| 1-RF | 38.59 | 20.21 | 13.76 | 10.57 | 8.04 | 6.53 | 5.85 | 5.56 |
| 2-RF | 7.55 | 6.99 | 6.70 | 6.53 | 6.38 | 6.31 | 6.27 | 6.26 |
| 3-RF | 7.45 | 7.20 | 7.06 | 6.97 | 6.89 | 6.85 | 6.82 | 6.81 |

**Floor-vs-defect split CONFIRMED** (pre-registered before 2/3-RF data). No single monotone winner — the ranking reshuffles with NFE through a **crossover cascade**: 3↔2-RF at ≈6 NFE, 1↔3-RF in (32,64), 1↔2-RF in (64,128). At NFE≥128 fully reversed to **1<2<3-RF** = published converged order (2.58<3.36<3.96). The **optimal reflow count is NFE-dependent** (3-RF at NFE=5, 2-RF on 8–64, 1-RF at ≥128). Reflow trades floor Q₀ for low-NFE defect d, exactly as the certificate predicts. Cross-method: the straightened path beats the best fixed-EDM solver only at NFE=5 (2/3-RF ≈7.5 vs dpm_v3 17.07, >2×); EDM+EMS's better floor (~4.35) wins at NFE≥8.

## 4. Schedule axis (m) — calibrated node placement on RF, *parameter-free*
Calibrate Euler defect `d(t)=‖ẍ(t)‖` on a 512-step reference (no FID feedback, no tuned scalar), place nodes `∝d^{1/(p+1)}`, p=1 (Euler order). Baseline = RF default uniform-Euler.

**On 1-RF the calibrated schedule beats uniform at every NFE:**

| K | 5 | 8 | 12 | 18 | 32 |
|---|----|----|----|----|----|
| uniform | 37.94 | 19.74 | 13.39 | 10.26 | 7.80 |
| proposed p=1 | 37.17 | 19.07 | 12.23 | 8.90 | 6.79 |

→ The certificate's **schedule prescription generalizes off the EDM family**, parameter-free, to a different ODE + solver. Sensitivity: optimal exponent mildly NFE-dependent (p=2 best at K=5,8; p=1 best at K≥12; p=0.5 over-concentrates) — echoes the EDM in-regime-exponent finding.

**FID-faithful weighting does NOT help (decomposability wall, cross-family).** Replacing pixel-`d` with `d·g` (g = feature-sensitivity, finite-difference) gives a *worse* schedule than both pixel-d and uniform at every NFE (48.4/26.3/16.9/11.6/8.0). g is well-behaved but peaks at the noise end (flow most sensitive to its initial condition), so `d·g` starves the high-curvature data end that needs resolution at deployment NFE. Same regime gap as EDM: the local/linear "principled" weighting is regime-mismatched; **parameter-free pixel-`d` is sufficient and superior.**

**Path×schedule interaction (full grid).** The calibrated schedule wins in *every* (path, NFE) cell, but the gap shrinks monotonically with reflow:

| ΔFID (proposed − uniform) | K=5 | K=8 | K=12 | K=18 | K=32 |
|---|----|----|----|----|----|
| 1-RF | −0.77 | −0.67 | −1.16 | −1.36 | −1.01 |
| 2-RF | −0.27 | −0.19 | −0.13 | −0.08 | −0.04 |
| 3-RF | −0.14 | −0.11 | −0.08 | −0.05 | −0.02 |

Mechanism (correcting a naive guess): reflow does **not** flatten the defect *shape* (max/median ≈ 69/71.8/72.1 for 1/2/3-RF, constant) — it shrinks the defect *magnitude*, so 2/3-RF run near their floors and leave little FID room for scheduling. Consistent with m* being scale-invariant in d (shape-driven leverage constant; absolute gain bounded by the defect's share of FID). **Path-straightening and schedule-calibration are complementary, not redundant.**

---

## Headlines
1. The generic certificate's structure holds on **all four axes** we could probe — and notably both the **path** (Q₀-vs-d crossover cascade) and **schedule** (parameter-free m*∝d^{1/(p+1)} beats default) claims **transfer to a new model family (Rectified Flow)**, not just EDM.
2. The schedule result is the strongest new generalization: a **parameter-free, FID-feedback-free** calibrated grid beats RF's uniform-Euler default at every NFE.
3. All pre-registered predictions were met; one naive sub-prediction (flatter defect on straighter paths) was honestly corrected by the data (shape preserved, magnitude shrinks).

## Caveats / honest scope
- All RF numbers are fixed-step Euler, 10k Clean-FID, 3 seeds; literature floors use adaptive RK45 (reproduced in rank, not absolute value, by our Euler floor probe).
- Schedule defect is **pixel-space** Euler truncation (‖ẍ‖), not FID-weighted; an FID-faithful weighting was *not* tested on RF (the EDM precedent showed it does not beat tuned, per the decomposability-wall finding).
- R2 inefficiency: recomputes the 128-step Heun reference per panel grid (~11 h); cache if rerun.

## Open / next
- OT-CFM path point (needs training, deferred). The reflow series already provides the controlled path-straightening ladder.
