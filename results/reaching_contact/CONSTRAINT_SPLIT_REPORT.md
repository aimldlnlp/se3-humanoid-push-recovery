# Frozen Normal and Tangential Constraint Controls

## Decision

Do not implement blanket normal/tangential constraint release as the next
controller correction. The tested controls do not provide enough support for it:
normal release only modestly reduces the lateral transition mismatch, tangential
release worsens it, and release markedly worsens correspondence in passing states.
This hypothesis is closed for direct promotion; no physical rollout or load
threshold is introduced. More adaptive contact policies are not disproved by
this audit, but are not validated by these results either.

The existing pose-origin candidate remains frozen and experimental. Its broader
disturbance gate stays four of eight new conditions. This audit does not make the
robot robust in all directions or qualify the hand-panel task.

## Method

Twenty-six frozen states from four prior trajectories: passing forward control,
passing diagonal 45 deg, failing lateral 90 deg and failing lateral 270 deg.
The selected experimental foot is right for the first three cases and left for
270 deg. Selection is declared per case, not computed by a tuned force threshold.

Four controls per state retain all point-kinematic equations, relax only normal
equations, relax only tangential equations, or relax all selected-foot equations.
For this horizontal ground plane, world z is normal and x/y are tangential;
ground orientation is checked. Only selected equation bounds become infinite.
The quadratic objective, linear objective term, matrix rows, force cones, wrench
synthesis, other foot, torque/acceleration limits, targets and gains stay fixed.
Original slack penalties remain; a slack disconnected from its equality can
minimize to zero. Lower objective cost is therefore not recovery evidence.

Applied push is restored for snapshot reconstruction, while the controller
force oracle stays disabled. Saved wrench and interval acceleration belong to
the original physical trajectory, not a counterfactual rollout. Constraint release
can allow force/motion combinations inconsistent with contact complementarity.
The controls are diagnostic feasible-set changes, not deployable contact modes.

## Transition Results

The metric below is the norm of predicted material-point acceleration minus the
saved observed acceleration over the following 4 ms interval, in m/s^2. Each
lateral transition has one selected contact point. Prediction and interval
observation are distinct quantities; closeness is not a causal recovery test.

| State / Selected Foot | Retain all | Release normal | Release tangential | Release all |
|---|---:|---:|---:|---:|
| 90 deg at 3.088 s / right | 7.376 | 6.569 | 8.335 | 6.243* |
| 270 deg at 3.096 s / left | 7.131 | 6.807 | 8.110 | 7.523 |
| PASS forward at 3.152 s / right | 0.217 | 3.062 | 23.631 | 29.758 |
| PASS diagonal at 3.152 s / right | 0.374 | 2.441 | 14.170 | 15.948 |

Normal release leaves a large lateral mismatch. Tangential release increases it
in both transition states. Release all is not consistently better, and its
apparent improvement in the 90 deg transition fails an additional cone check.
Passing-state rows retain their own point counts; norms concatenate components,
so comparison is within each state rather than between differing contact counts.

At the 90 deg transition, predicted selected-foot normal force is 11.004 N with
all constraints, 11.010 N with normal release, 3.259 N with tangential release,
and approximately zero with all release. At 270 deg it is 10.808, 10.714, 2.371
and 1.171 N respectively. Changed force distribution is part of each
counterfactual, not an observed new physical response.

Controls with no selected geometric contact relax zero rows and reproduce the
baseline. Empty-foot acceleration norms are zero by construction, not evidence
of a successful contact prediction. These controls remain explicitly identified.

## Numerical Qualifications

All 104 QPs are accepted by the unchanged row-wise numerical budget; maximum
ratio is 0.243. All 26 retain-all controls exactly reproduce saved controls and
wrenches. Maximum generalized acceleration is 35.187, below the original limit.
Unit tests check that only selected equality bounds change and that P, q, A
and all other bounds remain identical.

Two counterfactuals exceed the supplemental 1e-5 normalized inner-cone check:
90 deg / 3.088 s / release all (2.735e-5), and 270 deg / 3.056 s / release
tangential (1.367e-5). They remain recorded but are not counted as cone-qualified
witnesses. Row-budget acceptance and the stricter normalized diagnostic are
different tests. No force clipping, solver retuning or diagnostic threshold
relaxation is used to hide these cases. The starred table entry is one of them.

## Verification

105 tests pass. Original trajectory and output-manifest hashes match. Audit
source is `b17f36a`. No production controller, physical criteria, force cone,
gain or historical physical outcome changes. No new trajectory is generated.

[All controls](constraint_split_audit/audit.json),
[comparison metrics](constraint_split_audit/controls.csv),
[verification](constraint_split_verification.json).

## Milestone Boundary

Keep the demonstrated scope explicit: fixed-foot reaching at the tested target
and passing disturbance points, not a continuous force/direction envelope.
Reject this simple release rule instead of turning it into an unvalidated
adaptive controller. Reverse/80 N CoM failures remain a separate limitation.

A future panel milestone or a support-changing strategy needs its own explicit
scope and physical gates. Neither is authorized by an improved frozen residual
alone. Do not continue sweeping contact load thresholds or gains to rescue
this hypothesis. Reproduce in a fresh directory:

```bash
PYTHONPATH=src:experiments python experiments/reaching_constraint_split.py --output results/staging/constraint-split-new
```
