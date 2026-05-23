# Methodology — Autonomous Diffusion experiment factory

> The agent **autonomously improves on the validation split** and **evaluates once on
> the locked test split**. Run-until-success on the final metric is fake verification
> and is forbidden by the code path (see `src/autonomous_diffusion/critic/guards.py`).

## The pipeline

```
Theory  ->  Benchmark Design  ->  Implementation  ->  Validation Search
                                                              |
                                                              v
                              Locked Real-Image Test  <-  Freeze Best Config
                                          |
                                          v
                                 Paper Table / Figure
```

The seam between *Validation Search* and *Locked Real-Image Test* is the only
methodologically load-bearing seam. Once a config has been frozen and the
locked-test runner has started, the contract `locked_test_freeze:` fields cannot
change. Any attempt to change them aborts the run.

## Hard rules

1. **Validation vs test separation.** The validation split (10 000 samples) and
   the locked test split (50 000 samples) are loaded by different code paths and
   the test split is never touched by tuning code. The proposed method's
   hyperparameters are selected on validation only.
2. **One shot on test.** Each frozen config is evaluated on the locked test
   exactly once per declared seed. No "we tried it again with a small tweak"
   second evaluation.
3. **No baseline removal.** A baseline that beats the proposed method stays in
   the table. The reporter writes the loss; the critic refuses configs that drop
   any baseline listed in the contract.
4. **No metric switch.** The primary metric is `pareto_auc_fid_nfe`. If the
   proposed method loses on it, the report says so. We do not promote
   `clean_fid_at_matched_nfe` to primary post-hoc.
5. **No best-seed reporting.** All numbers are reported as mean ± SEM over the
   declared seed grid, with per-seed values saved alongside.
6. **Persist everything reproducible.** Every run writes `config_used.yaml`,
   `metrics.json`, `log.txt`, `env.json` (git commit + hostname + argv), and
   `generate_summary.json`. If we cannot reproduce a number, that number is
   unreported.
   - **Retention of `samples.npz`** is a separate axis from reproducibility.
     The default policy is `seed0_only`: seed-0 samples for every (sampler, NFE)
     are kept for inspection / qualitative figures; samples for seeds 1–4 are
     deleted after Clean-FID is computed. Set `RETENTION=keep_all` to override
     when disk allows; `RETENTION=delete_all` to keep only metadata + metrics.
     The retention action taken is recorded under `metrics.retention` per run.
7. **Sharp failure reporting.** If the theory fails on the locked test, the
   report carries the failure with the same prominence as a win. No quiet
   demotion to an appendix.

## How the critic enforces this

`src/autonomous_diffusion/critic/guards.py` exposes runtime asserts. The
locked-test runner (`src/autonomous_diffusion/experiments/run_locked_test.py`)
imports them and fails closed:

- `assert_locked_test_unchanged(contract, freeze_record)` — diff the contract
  fields under `locked_test_freeze:` against the recorded freeze snapshot.
- `forbid_test_split_in_validation(split)` — any code path tagged
  `phase=validation` that receives `split=test` raises.
- `forbid_baseline_removal(contract, run_plan)` — every baseline ID in the
  contract must appear in the locked-test plan.
- `forbid_primary_metric_change(contract, freeze_record)` — primary metric set
  must equal the freeze record's.
- `forbid_best_seed_reporting(metrics_record)` — refuse to emit a single-seed
  primary number.
- `check_seed_determinism(sampler, seed)` — same seed -> identical samples
  (max abs diff = 0). Run in `tests/` and at the head of every sweep.

These are runtime asserts because lint cannot see "we ran the test 7 times and
picked the best."

## What the agent IS allowed to do

- Iterate the proposed method's design on validation as many times as needed.
- Choose `epsilon`, `lambda`, `gamma`, NFE-grid points to search, and inner
  optimizer for u within the contract budget.
- Diagnose pipeline bugs (broken baseline, FID preprocessing mismatch,
  determinism violation) and re-run validation freely.
- Decide *when* to freeze, but not *which* fields to freeze (those are fixed by
  `locked_test_freeze:`).

## What the agent is NOT allowed to do

- Touch the locked test split during validation.
- Remove a baseline mid-sweep because it is winning.
- Change `metrics.primary` after the freeze.
- Report a hand-picked seed as the headline number.
- Mutate the generator (frozen pretrained EDM net) without bumping
  `model_policy.retrain_generator` in the contract and incrementing
  `contract_version`.
