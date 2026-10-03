# Contact Audit and Velocity-Damping Pilot

## Decision

Contact drift is a supported contributor, not a uniquely established root cause.
One opt-in damping candidate improves the moving-push case, but the hold-push
case still falls. The three-condition pilot gate fails. Do not promote the
candidate, sweep gains, or start the multi-target task yet.

## Audit

Ten existing trajectories are inspected: default/high-weight nominal and pushed
trials, plus the two guarded policies under moving/hold pushes. All target,
initial-state, time and force inputs match across their phase pairs within
`atol=1e-12`, allowing documented cross-platform roundoff. No new task
trajectories are generated for the audit; nominal setup is reconstructed only
to recover the original standing references.

Fifty frozen QP replays reproduce the stored actuator commands, with maximum
absolute difference below `8.2e-11` Nm. Logged arm references and effective reach
weights are restored for the guarded cases; the guard state machine is not
reinitialized at each snapshot. Applied force is not supplied to the controller.

| High-weight original condition | CoM band crossing | 25 mm foot drift crossing | First post-step contact loss | Torque bound | Fall height |
|---|---:|---:|---:|---:|---:|
| Moving push | 2.064 s | 2.344 s | 2.904 s | 3.596 s | 3.800 s |
| Hold push | 3.432 s | 3.668 s | 3.824 s | 4.448 s | 4.640 s |

The torso angular-speed band first crosses 20 ms after each pulse starts, also
in the successful default-weight moving trial. This response alone is not a
failure or a unique cause. Drift and CoM excursions subsequently precede
contact loss in the high-weight failures. Before first contact loss, peak torque
utilization is about 0.52 / 0.55, so actuator saturation and rejected QPs are not
the initiating observed events in those trials.

At 2.852 s, QP-predicted maximum foot-body linear acceleration is about
0.015 m/s^2 while the recorded next-interval average is 7.652 m/s^2. At 3.772 s,
the corresponding values are 0.020 and 7.106 m/s^2. Predicted contact slack is
small at these points, but actual body-foot motion has accumulated.

These are distinct quantities: instantaneous QP prediction versus a finite
difference of world-frame foot-body velocity over the following 4 ms. They are
not direct sole/contact-point accelerations. Foot-body rolling motion, compliant
contact and changing contact geometry can contribute; do not equate every
foot-body displacement with material-point sliding.

First contact crossings and sustained-contact-loss onset are reported
separately. Sustained durations use the original strict comparisons, without
roundoff relaxation. No historical success/failure classification is replaced
by the timeline's individual threshold crossings.

[Timeline CSV](audit/timeline.csv), [frozen replay](audit/audit.json),
[trajectory timeline](audit/contact_timeline.png),
[prediction versus actual](audit/prediction_vs_actual.png).

## One Candidate

The experimental controller changes only the contact acceleration right-hand
side from zero to `-20 * J_contact * qvel`, with matching bias diagnostics.
The existing slack penalty remains. Objective weights, torque/friction/support
limits, plant, target, pulse, and physical outcome thresholds are unchanged.
The 20 s^-1 coefficient is frozen for this exploratory pilot, not optimized.
It implies a nominal velocity decay time of 50 ms in the idealized model and
does not restore a foot-position reference.

| Front 12.5 cm, reach weight 3000 | Original | Damped candidate | Candidate hold max error | Candidate QP failures |
|---|---|---|---:|---:|
| No push | PASS | PASS | 3.234 mm | 0 |
| Moving, 70 N | FALL | PASS | 8.979 mm | 0 |
| Holding, 70 N | FALL | FALL | 814.523 mm | 1 |

All three fresh trials use local Windows CPU and exact paired input arrays.
The nominal candidate is not byte-identical to the original: hold error changes
from 3.231 to 3.234 mm, but remains well within the unchanged 15 mm gate.
The moving candidate keeps double support, foot displacement below 25 mm and
tangential speed below 0.08 m/s. The hold candidate delays first post-step
contact loss to 4.412 s, but still falls; delayed failure is not recovery.

No validation expansion or default promotion follows a 2/3 gate. The library
controller, default reach weight 100, and default contact behavior remain
unchanged. The option exists only in the reaching experiment.

[Pilot CSV](damping_pilot/study.csv), [pilot metadata](damping_pilot/study.json).
Scientific audit/pilot source is `c0b37546116549c0e2bc0c0d8e589900c28dc59a`.
Additional figures are regenerated from the same saved audit measurements.
87 tests pass locally. These trials record 100% misses of the 4 ms CPU deadline;
there is no hard-real-time or cross-host physical-replication claim.

## Next Gate

Follow-up completed: [hold geometry audit and pose-stabilization pilot](HOLD_REPORT.md).
The pose-restoration candidate passes only 1/3 conditions and is not promoted.
The following describes the original audit's next gate, retained for chronology.

Replay the hold candidate before its slip onset and compare foot roll/contact
geometry, position drift and CoM response with the original held-target state.
Use that evidence to decide whether position stabilization or a contact-model
correction is warranted. Do not add both, tune on a validation set, or loosen
the success criteria. Multi-target reaching waits until nominal, moving-push,
and hold-push gates all pass.
