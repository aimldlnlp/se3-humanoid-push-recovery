# Frozen Pose-Component Feasibility Audit

## Decision

Keep the contact model and CoM gains unchanged. Preserving the attained torso
and pelvis outputs together restricts instantaneous CoM correction. At the last
pre-excursion snapshot, the torso combination is more restrictive than the
pelvis combination; the angular pair is more restrictive than the linear pair.
No single component alone prevents exact CoM correction in that snapshot.
These results do not justify disabling torso control or lowering rotation gains.

## Method

Replay the same eight saved states from 2.996 to 3.436 s. All replayed controls
match exactly. Preserve attained reaching acceleration in every control, then
test all 16 subsets of torso linear/angular and pelvis linear/angular outputs:
128 controls total. Original QP constraints and accepted point slack remain
fixed. First minimize CoM L1 acceleration residual, then minimize peak generalized
acceleration within 1e-9 of that optimum. This second objective avoids arbitrary
extreme LP solutions without changing the existing acceleration bound.

Linear/angular mean rows of the implemented SE(3) task output. The pose target
uses transported gains and an SE(3) logarithm, so these rows are not independent
position-gain and rotation-gain contributions. Preserved values are attained
outputs, not desired task targets. Peak generalized acceleration mixes linear
and angular units, matching the original componentwise bound.

## Last Pre-Excursion Snapshot

At 3.436 s, CoM displacement is 99.936 mm; the 100 mm band crosses at 3.440 s.
Reaching is preserved in every row below.

| Additional attained outputs preserved | Minimum CoM L1 residual (m/s^2) | Minimum peak generalized acceleration |
|---|---:|---:|
| None | 0 | 58.694 |
| Full torso | 1.876319 | 150 |
| Full pelvis | 0.797209 | 150 |
| Torso linear + pelvis linear | 0.912372 | 150 |
| Torso angular + pelvis angular | 2.001074 | 150 |
| Full torso + full pelvis | 2.095132 | 150 |

Preserving each component alone still permits zero CoM residual. Minimum peak
accelerations are 87.000 for torso linear, 139.905 for torso angular, 111.641
for pelvis linear, and 117.026 for pelvis angular. Thus conflict depends on
combinations, and angular preservation can consume more acceleration authority.
Pointwise feasibility is not proof of a stable rollout.

The actual QP weighted costs at this snapshot are 16,531.920 for torso linear,
478.506 for torso angular, 1,749.736 for pelvis linear, and 1.262 for pelvis
angular. Each pair reconstructs its original pose-task cost. Large linear cost
does not establish that the linear component is the dominant feasibility conflict.

## Numerical Limits and Verification

112 first-stage controls solve. All 16 exact fixed-slack controls at 3.080 s
report infeasible, consistent with the earlier audit. The original replayed QP
is accepted within its unchanged numerical budget. These stricter diagnostic
equalities do not establish physical infeasibility.

111 controls also produce validated minimum-acceleration witnesses. Three
secondary solves initially return `Unknown`; one retry with HiGHS IPM on the
same LP resolves two. Full torso + pelvis preservation at 3.020 s remains
`Unknown` in the secondary solve. Its first-stage residual is 0.166582 m/s^2,
but no minimum-acceleration witness is claimed. All statuses remain in the data.

All validated witnesses satisfy the original constraint budget and acceleration
bound. Saved trajectory and artifact hashes match. 96 tests pass. Audit source
is `2f8279a`; no production controller, reference, gain, contact equation or
physical success threshold changes. No new physical trial runs. The existing
nominal/moving/hold physical gate remains 2/3, with held-target FALL.

[Controls](pose_split_audit/controls.csv),
[component costs](pose_split_audit/component_costs.csv),
[complete audit](pose_split_audit/audit.json),
[verification](pose_split_verification.json).

## Next Gate

Completed by the [pose convention audit](FRAME_REPORT.md). It confirms an origin
mismatch in the production pose inputs but does not change the controller or
establish that correcting it will recover the held-target trial.

Verify SE(3) target and Jacobian frame/origin consistency before choosing one
reference or priority correction. Output-component conflicts alone do not prove
a gain or coordinate-convention defect. Any later controller change must rerun
the unchanged nominal, moving-push and held-target physical gates.

Reproduce in a fresh directory:

```bash
PYTHONPATH=src:experiments python experiments/reaching_pose_split_audit.py --output results/staging/pose-split-new
```
