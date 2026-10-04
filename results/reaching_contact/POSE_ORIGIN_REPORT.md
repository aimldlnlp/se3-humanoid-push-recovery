# Body-Origin Pose Correction Pilot

## Decision

The opt-in pose-origin candidate passes all three deterministic simulation gates:
nominal, moving push and held-target push. The prior contact-mode candidate
passed only nominal and moving; hold was FALL. Keep the correction experimental:
one paired pilot is not held-out validation or a production promotion.

No gain search, threshold relaxation, contact-model change, stepping or payload
is introduced. The library controller and existing historical artifacts remain
unchanged. Source is `a607d70`.

## Correction

Evaluate the existing SE(3) pose law using the world-origin spatial velocity,
then convert its acceleration target to the world-aligned body-origin convention.
Include moving-origin and geometric Jacobian derivative terms:

```text
H(p) = [[I, hat(p)], [0, I]]
spatial_target = existing_pose_law(T, T_des, H J, qvel, same_gains)
b_geometric = Jdot qvel
offset = [linear_velocity cross angular_velocity; 0]
target_for_J_qdd = inverse(H) (spatial_target - offset) - b_geometric
```

The objective retains the geometric Jacobian and its original scalar weight.
Thus its residual metric stays at the body origin; it is not an identity metric
on world-origin spatial twists. Tests verify the complete pose quadratic
(matrix, linear term and constant) under a translated/rotated world frame, plus
the physical/spatial acceleration bookkeeping. The nonlinear SE(3) resolved
acceleration law remains approximate.

## Frozen QP Gate

The eight original pre-excursion states are replayed with baseline and candidate.
All sixteen QPs satisfy the original numerical budget. Baseline controls match
the saved trajectory. Constraints, quadratic matrix and every non-pose objective
are exactly identical; only torso/pelvis targets change. Objective reconstruction
is checked against each full QP.

At 3.436 s, CoM residual is 4.709 m/s^2 for baseline and approximately 4.757 for
the candidate. Reaching residual also increases at this frozen state. Therefore
the frozen comparison does not itself predict better recovery. Candidate pose
costs use different targets and are not improvement scores against the baseline
targets. The later physical result comes from a different closed-loop trajectory.

## Paired Simulation Gate

Front reaching remains 12.5 cm, weight 3000. Contact-mode equations, damping
20/s, gains, 5 s duration, initial condition and physical criteria are unchanged.
Push is 70 N for 0.15 s, starting at 1.5 s while moving or 3.0 s while holding.
The nominal condition has no push. The final 0.5 s hand-error tolerance is 15 mm.
Pairs match time, reference, goal, push and initial state arrays exactly.
MuJoCo, NumPy and OSQP versions match the prior contact-mode baseline.

| Condition | Previous result | Corrected result | Maximum final-hold hand error | Maximum foot drift |
|---|---|---|---:|---:|
| Nominal | PASS | PASS | 4.010 mm | 0.426 mm |
| Moving push | PASS | PASS | 3.768 mm | 3.784 mm |
| Held-target push | FALL | PASS | 4.804 mm | 6.303 mm |

All corrected trials have continuous double support and zero QP failures.
Held-target recovery reports maximum CoM displacement 96.033 mm, maximum torso
error 0.054292 rad, peak torque 21.988 Nm and recovery latency 0.258 s. Whole-trial
assessment and final balance checks also pass; this is not only a hand-error gate.

Local deadline misses are 100% in all three corrected trials, as in the prior
contact-mode pilot. The simulator advances fixed timesteps, but this Python
experimental implementation is not demonstrated to meet the real-time deadline.

## Verification and Limits

100 tests pass. Frozen trajectory hashes, paired inputs and all output manifests
are verified. The correction combines origin consistency with acceleration-bias
compensation in one candidate; this pilot does not isolate their causal effects.
No alternate gains or unsuccessful trials were discarded. No hardware runs occur.

[Frozen comparison](pose_origin_audit/audit.json),
[simulation study](pose_origin_pilot/study.json),
[verification](pose_origin_verification.json).

## Next Gate

Freeze this candidate. Validate at previously untested push directions and nearby
push magnitudes/timings, keeping all gains and criteria fixed. Report failures
and the small held-target CoM margin, not only successes. Separately measure and
reduce solver/runtime overhead before claiming real-time feasibility. Do not
expand reaching distance or promote the production default from this pilot alone.

```bash
PYTHONPATH=src:experiments python experiments/reaching_pose_origin.py --output results/staging/pose-origin-audit-new
PYTHONPATH=src:experiments python experiments/reaching_contact_pilot.py --contact-mode --pose-origin --output results/staging/pose-origin-pilot-new
```
