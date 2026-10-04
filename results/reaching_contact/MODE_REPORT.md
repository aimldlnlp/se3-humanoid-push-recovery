# Matching Contact Wrench and Motion Prototype

## Decision

The contact formulation now uses the same reconstructed material points for
wrench capability and acceleration constraints. Nine frozen counterfactual
states pass the numerical/cone gate. The physical pilot remains 2/3: nominal
and moving push pass, held-target push falls. Keep the option experimental and
disabled by default. No gain sweep, priority change or multi-target expansion.

## Formulation

The prototype supports the observed `condim=3` contacts only. Each geometric
ground-contact point provides nonnegative coefficients on four inner diamond
friction rays. Net wrench about the body origin equals the sum of those point
forces and their lever-arm moments. A point cannot provide an independent
spin or rolling moment. Original world L1 friction/normal bounds are retained;
static rectangular-patch moment bounds are replaced by point-force synthesis.

The same body-fixed material points supply translational Jacobians and exact
MuJoCo `Jdot*qvel`. Their acceleration targets use damping 20 s^-1 on point
velocity. No separate angular foot-body constraint is retained. The rigid-body
motion-constraint rank is six for a non-collinear flat patch, five for an edge,
and three for a point. Rotation about an edge or point is permitted by geometry,
not by weakening an arbitrary angular gain. With no points the foot wrench
is zero and no point kinematic constraints are added.

The original twelve body-slack variables remain in the base layout but are
fixed to zero; point-slack variables are appended. The original scalar penalty
is averaged over points per foot. **The slack metric changes** from combined
linear/angular body deviation to point acceleration deviation. This is not an
otherwise-identical objective ablation. Torso, pelvis, CoM, posture, reaching
and nominal-torque objective coefficients remain unchanged.

The experimental result's contact residual/slack fields report point target
deviation before slack. The solver's unchanged row-wise budget check separately
validates augmented constraint equalities. These diagnostics are not directly
interchangeable with historical body-contact residual norms.

## Frozen Gate

Five hold and four moving snapshots from the damping-only trajectories are
replayed with the old controller and then solved counterfactually with the
prototype. Baseline controls match the recordings. All nine prototype QPs are
accepted by the original numerical budget; the largest budget ratio is 0.030243
(acceptance limit 1). Each predicted wrench fits the enlarged reconstructed
outer cone at normalized residual no greater than `1e-5`. Measured original
wrenches fit the same outer cone as positive controls.

This only permits a physical pilot. Outer-cone acceptance is not proof of
physical feasibility, and baseline next-interval accelerations in the
counterfactual records are still baseline observations, not prototype rollouts.

At the difficult 4.284 s hold state, both motion ranks are three. CoM task
acceleration residual is 22.300 m/s^2 for the baseline and 22.735 m/s^2 for the
prototype. Reaching residual is 0.982 versus 1.676 m/s^2. Fixing geometry does
not make this already-diverged state recoverable or realize the CoM task.

[Frozen records](mode_audit/audit.json).

## Physical Pilot

All three fresh simulations use the original initial state, front target
12.5 cm, reach weight 3000 and pulse inputs paired to damping-only baselines.

| Condition | Damping baseline | Prototype | Hold max error | Max foot XY displacement | QP failures |
|---|---|---|---:|---:|---:|
| No push | PASS | PASS | 3.236 mm | 0.425 mm | 0 |
| Moving, 70 N at 1.5 s | PASS | PASS | 2.961 mm | 7.938 mm | 0 |
| Hold, 70 N at 3.0 s | FALL | FALL | 317.133 mm | 98.607 mm | 0 |

Moving hold error decreases from 8.979 to 2.961 mm and maximum foot displacement
from 22.165 to 7.938 mm. Tangential speed stays below 0.08 m/s at 0.063537 m/s;
double support remains throughout. Motion ranks switch between five and six.
Nominal arrays are not identical to the damping baseline, but the unchanged
15 mm hold gate remains satisfied.

Hold CoM crosses the 100 mm band at 3.440 s, foot drift crosses 25 mm at
4.316 s, first contact loss occurs at 4.348 s and fall height at 4.588 s. Torque
bound crossing at 4.568 s comes later than drift/contact loss. No QP is rejected.
Error reduction in an eventual fall is not recovery. No default promotion or
success reclassification follows a 2/3 gate.

Contact point counts and motion ranks are stored per step in each NPZ. Rank
zero means no geometric contacts; rank -1 indicates an unsupported/build-failed
mode, explicitly accompanied by QP fallback. The recorded pilot contains no
rank -1 states. Rank counts do not establish which contacts actually carry load.

[Pilot CSV](mode_pilot/study.csv), [pilot metadata](mode_pilot/study.json).

## CoM Task Controls

The previous pose-restoration moving TIMEOUT is also replayed as a separate
control, not combined with the prototype. At 4.500 / 4.900 s, CoM acceleration
target residual norms are 3.883 / 3.264 m/s^2, while reaching residuals are
0.448 / 0.357 m/s^2. These compare `J*qdd` to each implemented task target;
they are not physical next-interval acceleration errors. Different task weights
and feasibility constraints make raw norms insufficient to assign one culprit.

[Task-control snapshots](mode_audit/timeout_task_residuals.json).

## Verification and Limits

95 tests pass locally. New tests cover point/edge rotational freedom, flat
rank, force-to-origin moment mapping, accepted augmented QP, and observable
fallback for unsupported modes. All three physical outcomes are independently
reclassified and their input pairs verified. Manifest hashes are checked.
Defaults, physical thresholds, library controller and main branch are unchanged.

Frozen prototype scientific source: `1841660`. Physical source: `8bca4ec`.
Unsupported-mode logging is hardened afterward at `b0d2b93`; it does not change
the supported `condim=3` pilot equations. Physics is local Windows CPU, without
cross-host replication. Every pilot misses the 4 ms deadline; no real-time claim.

The prototype uses instantaneous geometric contacts, including possibly unloaded
ones. It has no load-bearing estimator, transition hysteresis, compliant-contact
model, touchdown policy or support for `condim` above three. Abrupt topology and
friction-ray changes remain a limitation. The graph's older contact/Jacobian
references were checked against current source; the graph was not rebuilt.

## Next Gate

Follow-up completed: [pre-excursion task and constraint audit](TASK_REPORT.md).
Contact model and task weights remain frozen. Original next-gate rationale follows.

Inspect the new hold trajectory before the 3.440 s CoM excursion. Quantify
weighted torso, pelvis, posture, reach and CoM objective residuals and active
constraints at the same states. Use a frozen counterfactual to isolate whether
priorities or feasible support limits block restoration before altering one.
Keep this contact model fixed for that investigation. Do not add another contact
gain, relax the recovery bands, or call a later fall an improvement.

Reproduce with fresh output directories:

```bash
PYTHONPATH=src:experiments python experiments/reaching_contact_mode_audit.py --output results/staging/mode-audit-new
PYTHONPATH=src:experiments python experiments/reaching_contact_pilot.py --contact-mode --output results/staging/mode-pilot-new
```
