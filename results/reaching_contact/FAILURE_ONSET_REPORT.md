# Failure Onset and Contact Prediction Audit

## Decision

The rejected disturbance cases do not share one onset sequence. Lateral cases
lose geometric support before slip; reverse/strong forward pushes exceed the
CoM band while both feet still have contact. Do not choose one blanket gain or
friction-cone correction from these failures. No controller or new trajectory
changes in this audit. The broader gate remains four of eight new conditions.

## Scope and Time Alignment

Forty saved-state QPs across six cases: four failures, the reproduced forward
positive control and the passing diagonal condition. Controls and predicted
wrenches replay with zero maximum difference. Applied push force is restored
before replay; the controller's force oracle remains disabled.

Saved pre-step wrenches are compared with predictions at the same body origins.
Post-step channels are separately labeled and timestamped at the control
interval end, usually 4 ms later than the row's pre-step time. Material points
are fixed in each foot's local coordinates when comparing velocities across the
next saved-state interval. They need not remain in contact throughout that
interval. Instantaneous QP acceleration and interval-average observed acceleration
are not the same mathematical quantity.

Event extraction reports first crossings and starts of qualifying sustained
runs after push start. This is not a reimplementation of classifier priority or
its post-push evaluation window. A sustained onset is backdated to the run start,
not reported at the later confirmation time. The 100 mm CoM band, 25 mm foot
drift, 0.08 m/s tangent speed, 0.04 s slip duration and 0.08 s contact-loss
duration remain unchanged.

## Observed Ordering

| Case | First contact loss | First speed crossing | Sustained slip onset | CoM band crossing | Fall-height crossing |
|---|---:|---:|---:|---:|---:|
| 90 deg | 3.092 s | 3.188 s | 3.224 s | None | None |
| 270 deg | 3.100 s | 3.276 s | 3.308 s | None | None |
| 180 deg | 3.980 s | 4.092 s | 4.184 s | 3.572 s | 4.224 s |
| 80 N | 4.160 s | 4.284 s | 4.112 s | 3.392 s | 4.396 s |

The 80 N sustained slip starts from displacement crossing, not tangent speed,
which explains its earlier time. The first lateral contact losses can be brief:
90 deg never meets the sustained contact-loss duration. At 270 deg the qualifying
contact-loss run begins at 3.120 s. The positive control and diagonal condition
have none of the listed crossings after the push begins.

Ordering narrows where to inspect; it does not establish failure causality.

## Lateral Transition

At 3.088 s in the 90 deg case, the right foot has one geometric contact (rank 3).
QP predicts vertical force 11.004 N; saved actual force is 9.978 N. Predicted
point acceleration differs from the damping target by only 0.000928 m/s^2.
The next interval's observed material-point acceleration differs from prediction
by 7.376 m/s^2. That interval ends at the first contact loss, 3.092 s.

At 3.096 s in the 270 deg case, the left foot similarly has one contact, with
predicted/actual vertical force 10.808/9.512 N. QP point-target residual is
0.000530 m/s^2; observed interval difference is 7.131 m/s^2. Its first contact
loss is at the interval end, 3.100 s.

By 3.152 s the unloaded foot has no geometric contacts in both lateral cases;
its QP and actual forces are essentially zero. The remaining foot carries the
load. Thus the audit does not show a QP assigning a large force to a completely
absent contact. It shows a discrepancy in predicted material-point motion near
a low-load contact transition. Contact compliance, changing support and the
unmodeled external disturbance are not isolated by this snapshot comparison.

The passing diagonal case also has a single-contact low-load foot at 3.152 s,
but actual vertical force is 37.010 N and interval acceleration difference is
0.374 m/s^2. A contact rank change alone is not a failure criterion.

## Force Feasibility and Non-Lateral Cases

All 80 saved measured foot wrenches fit both reconstructed inner and outer cones
with zero reported residual. Maximum predicted inner-cone residual is 4.46e-6
under the fixed force/moment normalization, within 1e-5. Cone membership is not
proof of no-slip, sticking kinematics or force realizability over time. These
samples do not support a simple cone-capacity mismatch as the whole explanation.

In the 80 N case at the first CoM crossing, 3.392 s, both feet retain two-point
edge contacts. Force prediction errors are only 2.424 and 2.193 N; interval
point-acceleration differences are 0.035 and 0.030 m/s^2. Its large later drift
must not be used to label contact loss as the initiating event.

The reverse-push case is different again: at 3.152 s the left-foot force
prediction error is 89.180 N. At its CoM crossing, 3.572 s, both feet have three
contacts (rank 6) and force errors are 15.561 and 20.157 N. This warrants task
and force-distribution inspection before later support loss, not an assumption
that the lateral low-load mechanism explains every failure.

During the push, the non-oracle QP intentionally excludes its applied force.
The report therefore stores force-based CoM acceleration both without that
force and with it included diagnostically. The added known force is not fed
back to the controller. Saved measured-force acceleration and observed next-
interval acceleration remain distinct; other non-foot contacts can invalidate
foot-only force balance after a fall. Selected snapshots precede fall height.

## Verification

103 tests pass, including sustained-run segmentation and post-step timestamp
tests. All 40 QPs pass the unchanged numerical budget (maximum ratio 0.204).
All original trajectory hashes and audit manifest hashes match. Audit source is
`37ff86f`; additional clock tests are committed in `c8b949e`. No gain, contact
equation, threshold, production controller or physical outcome is changed.

[Events and snapshots](failure_onset_audit/audit.json),
[contact measurements](failure_onset_audit/contacts.csv),
[verification](failure_onset_verification.json).

## Next Gate

Completed by the [frozen constraint split](CONSTRAINT_SPLIT_REPORT.md). Simple
normal/tangential release is not selected for controller correction. The tested
controls do not justify a rollout policy or a load threshold.

First isolate the low-load lateral transition with frozen counterfactuals:
separate normal versus tangential point constraints and the sticking assumption
while preserving the pose/reaching targets and existing force cones. Evaluate
accepted accelerations against the original observed transition; do not turn
these controls into a new policy or select load thresholds from two snapshots.

Separately inspect body-task compromise and predicted force distribution before
CoM drift in the reverse/80 N cases. These are two diagnostic branches, not
permission for a gain sweep. Panel contact and production promotion remain
deferred. Reproduce in a fresh directory:

```bash
PYTHONPATH=src:experiments python experiments/reaching_failure_onset.py --output results/staging/failure-onset-new
```
