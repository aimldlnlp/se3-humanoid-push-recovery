# Front-Reaching Objective Conflict Audit

## Question

Does removing the posture acceleration objective or nominal-torque objective
resolve the front 12.5 cm failure under a 70 N forward push at reach weight 3000?

## Method

- Nine new local CPU trials: remove posture, remove nominal torque, or remove
  both; each with no push, a moving push at 1.5 s, and a hold push at 3.0 s.
- Three original baseline trajectories are copied byte-identically from the
  previous calibration. Their provenance is retained, not retagged.
- All inputs, initial states, Cartesian references, force traces and physical
  success thresholds are unchanged. Only the named soft objectives change.
- Eight original states are replayed before contact loss. Four QPs per state
  provide 32 frozen-state evaluations. Original controls match recorded
  controls within `rtol=1e-7, atol=1e-7`; physical constraint matrices/bounds
  hash identically across the four modes at each state.
- Weighted objective costs and reach/posture gradient cosines describe
  optimization sensitivity, not an independent proof of physical causation.
  Counterfactual frozen solves do not generate new measured contact forces.

## Findings

All four modes pass without push. Maximum final-hold error is 3.23 mm for
the original, 2.47 mm without posture, 3.28 mm without nominal torque, and
2.50 mm without both objectives. **Every pushed trial falls**, including all
six new ablations. Lower final tracking error in a fallen trajectory is not
success; the whole-task physical test remains mandatory.

First sampled loss of either physical foot contact:

| Mode | Moving push | Hold push |
|---|---:|---:|
| Original | 2.904 s | 3.824 s |
| No posture objective | 2.432 s | 3.688 s |
| No nominal-torque objective | 2.080 s | 3.652 s |
| Neither objective | 2.360 s | 3.432 s |

These are first-sample event times, not the classifier's sustained-contact
loss times. In the original moving trial, contact loss precedes torque-bound
contact at 3.596 s, fall-height crossing at 3.800 s, and QP failure at 4.228 s.
The original hold trial loses contact at 3.824 s, reaches a torque bound at
4.448 s, crosses fall height at 4.640 s, and has no QP failure.

At the last sampled pre-loss hold state (3.772 s), predicted normal forces
are 93.59/219.16 N, while measured normal forces are 147.32/216.22 N. This
documents a plant/predicted-contact discrepancy; it does not identify a
unique cause. The controller still assumes fixed double support.

## Decision

Do not delete either objective or promote the weight-3000 profile. These
deletions do not fix the tested failures and move contact loss earlier.
Torque saturation and solver failure are not the initiating observed events
in the original trials. The audit does not rule out interaction among tasks,
nor establish a unique root cause or general robustness result.

Next targeted experiment: bound the influence of reaching using measured
balance/contact state, before first contact loss, while retaining the original
stabilizing objectives. Keep a matched no-push tracking gate and paired push
controls; do not add stepping or relax physical thresholds.

## Provenance

New ablation/audit source: `04bf33403299ba8cb6313cf081dfb3b984d4cea5`.
Reused baselines: `fbc0b509402bc8d60a365c0b612d331a120fdc88`.
81 tests pass locally and on the SSH worker. The worker source snapshot was
completed with the repository's bundled fonts before the successful retry.
Legacy `scp -O` succeeds where the default transfer repeatedly reset.
Physics runs are local; worker tests are not worker trajectory replication.
Local 4 ms deadline misses remain 100%, so no hard-real-time claim is made.

Raw results: `study.json`, `study.csv`, per-trial summaries and NPZ files.
Verification: `verification.json` and `manifest.json`.
