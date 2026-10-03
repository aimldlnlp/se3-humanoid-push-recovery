# Hold-Push Geometry Audit and Pose-Stabilization Pilot

## Decision

Adding foot-pose restoration alone does not solve held-target recovery. Keep
both contact corrections experimental and disabled by default. Do not expand
to multi-target reaching or search more gains from these three conditions.

## Saved-State Audit

Two existing hold trajectories are inspected: original reach weight 3000 and
velocity damping 20 s^-1. Their initial states, target, force and time arrays
are paired. Ten frozen QP replays reproduce the saved controls with zero
reported difference on the local runtime. No new physical trials are run for
this audit.

Before the damped candidate's first slip-speed crossing at 4.336 s, the
4.284 s snapshot shows:

- Left/right body-origin XY drift: 9.142 / 9.362 mm.
- Left/right sole tilt from world vertical: 3.505 / 6.950 degrees.
- One reconstructed ground-contact point per foot, versus four before push.
- Logged pre-step normal forces: 166.500 / 68.758 N.
- Measured post-step double support remains true.

The original candidate at 3.792 s has 46.048 / 45.111 mm body-origin drift
and 37.708 / 36.853 degrees tilt. The damped candidate reduces this roll but
does not restore the initial pose. This supports testing pose restoration;
it does not prove that missing position feedback is the unique root cause.

The reconstructed manifold uses saved pre-step qpos/qvel and MuJoCo forward
kinematics. Point counts do not establish which contacts carry force. Normal
forces and contact-point slip speeds remain original logged measurements, with
pre/post-step timing explicitly distinguished. Body-origin drift is not a
direct measurement of material-point sliding. Sole corners use the existing
body-local support profile, not a new collision shape.

[Geometry CSV](hold_audit/geometry.csv), [frozen replays](hold_audit/audit.json),
[pose and load figure](hold_audit/hold_geometry.png).

## One Frozen Candidate

The experiment changes only the contact acceleration target:

```text
a_contact = -20 * velocity_world - 100 * pose_error_world
```

Foot references are captured from the initial settled stance and never moved
with the current foot. Position error is world translation; orientation error
is the SO(3) logarithm of current rotation times reference rotation transpose,
matching the world-frame body Jacobian. Stiffness 100 s^-2 and damping 20 s^-1
form an ideal critically damped scalar pair; this is not a stability guarantee
for the constrained nonlinear plant. No gain sweep is performed.

Reach weight 3000, front offset 12.5 cm, 70 N pulses lasting 0.15 s, objective
weights, contact model, physical limits and all success criteria are unchanged.
The option defaults to zero and lives only in the reaching experiment.

| Condition | Damping only | Pose restoration | Hold max error | Peak foot XY displacement | QP failures |
|---|---|---|---:|---:|---:|
| No push | PASS | PASS | 3.238 mm | 0.425 mm | 0 |
| Moving push, 1.5 s | PASS | TIMEOUT | 5.706 mm | 2.639 mm | 0 |
| Hold push, 3.0 s | FALL | FALL | 966.136 mm | 261.243 mm | 1 |

Gate: **1/3 combined passes, failed**. Moving contact drift improves, but
final hold CoM displacement reaches 114.526 mm, beyond the unchanged 100 mm
balance band. Torso error stays below 1.341 degrees, angular speed below
0.056 rad/s, double support remains, and there are no slip/contact-loss events
or rejected QPs. Good hand tracking and small foot drift are not sufficient
for recovery. Extending the observation window would define a different test;
it is not used to relabel this TIMEOUT.

The hold candidate's first contact loss is 4.248 s, versus 4.412 s for damping
alone. Fall height is crossed at 4.540 s. Pose restoration does not delay all
failure signals and is not an improvement overall.

[Pilot CSV](pose_pilot/study.csv), [metadata](pose_pilot/study.json).

## Verification and Limits

- 89 local tests pass, including world-frame pose correction, fixed references,
  contact RHS/bias consistency, unchanged objective matrices, and zero damping.
- All three new outcomes are independently reclassified from saved arrays.
- All three input pairs match the damping baseline.
- All 21 files in the two new run manifests pass SHA-256 checks.
- Audit scientific source: `91a0443`; pilot scientific source: `98e02e4`.
- Audit and physics run locally on Windows CPU; no cross-host replication.
- All three new trials record 100% misses of the 4 ms deadline. No real-time claim.
- Library controller, defaults, physical thresholds and main branch are unchanged.
- Existing graph vocabulary guided source discovery; its older line references
  were cross-checked against current code rather than treated as current truth.

## Next Gate

Audit predicted versus measured contact wrenches against the *current* contact
geometry, especially when a foot changes from a flat patch to an edge or point.
The QP currently retains horizontal rectangular-patch CoP limits while these
saved states show tilted soles and fewer geometric contact points. Check
whether its requested moments remain realizable before adding a contact-mode
correction. Separately inspect why CoM stays displaced in the moving candidate
despite constrained feet. No new controller change is justified as proven by
this audit alone; no automatic promotion, gain sweep, or threshold relaxation.

Reproduce the follow-up from this branch:

```bash
PYTHONPATH=src:experiments python experiments/reaching_hold_audit.py --output results/staging/hold-audit-new
PYTHONPATH=src:experiments python experiments/reaching_contact_pilot.py --pose-stabilization --output results/staging/pose-pilot-new
```

Output directories must be fresh. Original trajectory bundles are preserved.
