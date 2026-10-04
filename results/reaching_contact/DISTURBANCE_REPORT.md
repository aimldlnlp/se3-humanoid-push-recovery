# Frozen Pose-Origin Disturbance Validation

## Decision

The broader disturbance gate fails. The repeated positive control passes, but
only four of eight new conditions pass unchanged criteria. The earlier 3/3
pilot was specific to the tested forward push, not general push robustness.
Keep the candidate experimental; defer the proposed hand-panel contact task.
Do not retune gains, weaken slip limits or promote the production default.

## Predeclared Scope

Nine deterministic 5 s simulations: one positive control and eight single-factor
changes. The plan was fixed before outcomes, recorded in `plan.json`, and all
conditions completed even after failures. Reaching stays 12.5 cm forward, weight
3000. Contact-mode equations, pose-origin correction, damping 20/s, gains and
physical criteria are unchanged. Push duration remains 0.15 s. Unless indicated,
magnitude is 70 N, direction is 0 degrees and start time is 3.0 s.

Directions are horizontal force vectors in world coordinates: 0 is +x, 90 is +y,
180 is -x and 270 is -y. These are not rotation commands or reaching directions.
This is a small deterministic validation, not a probabilistic success estimate,
an exhaustive disturbance boundary or hardware validation.

## Results

| Condition | Result | Maximum final-hold hand error (mm) | Maximum CoM XY displacement (mm) | Maximum foot drift (mm) |
|---|---|---:|---:|---:|
| Positive control, 70 N / 0 deg / 3.0 s | PASS | 4.804 | 96.033 | 6.303 |
| Direction 45 deg | PASS | 3.979 | 52.864 | 3.342 |
| Direction 90 deg | SLIP | 4.098 | 29.741 | 29.467 |
| Direction 180 deg | FALL | 257.306 | 666.264 | 1277.446 |
| Direction 270 deg | CONTACT_LOSS | 4.672 | 44.363 | 57.513 |
| Magnitude 60 N | PASS | 4.009 | 62.795 | 3.294 |
| Magnitude 80 N | FALL | 359.494 | 891.169 | 1474.363 |
| Start at 2.75 s | PASS | 3.825 | 95.860 | 6.299 |
| Start at 3.25 s | PASS | 8.628 | 96.217 | 6.312 |

There are zero QP failures in every trial. Successful optimization is therefore
not sufficient for physical balance, no-slip or support retention. At 90 deg,
hand error passes but foot drift is 29.467 mm and peak tangent speed is 0.310 m/s.
At 270 deg, hand error also passes while support is lost. The full reaching and
balance assessment correctly rejects both cases.

Failure reasons above follow the existing classifier's priority. They are not
exclusive violations or diagnoses of the first causal event. For example, the
270 deg trial also exceeds slip measurements, while CONTACT_LOSS is the reported
primary category. Large post-fall drift is not evidence of the initiating cause.

The 80 N failure contrasts with the 70 N positive control and 60 N success.
This brackets tested outcomes for this direction/timing, not an exact force
limit or a monotonic recovery guarantee. Successful timing variants retain a
small CoM margin to the unchanged 100 mm band.

## Verification

101 tests pass. The positive control reproduces the previous corrected hold
dynamics. Every trial matches the reference initial condition, reaching
references, goal, weight history, model hash, dependency versions and settings.
The only varied input is the predeclared push factor. Saved force arrays match
the specified direction, magnitude and pulse timing, including existing boundary
tolerance. Realized impulse is checked against magnitude times 0.15 s.

Candidate implementation is still `a607d70`; validation source is `61b20c0`.
No library source, robot model, configuration, pose-origin controller or contact
model changes relative to the corrected pilot. All failures and raw trajectories
are retained. Local deadline misses remain 100%, so real-time feasibility is
still not demonstrated. No hardware trial or default promotion occurs.

[Predeclared plan](pose_disturbance_validation/plan.json),
[complete study](pose_disturbance_validation/study.json),
[trial metrics](pose_disturbance_validation/study.csv),
[verification](disturbance_verification.json).

## Next Gate

Completed by the [failure onset audit](FAILURE_ONSET_REPORT.md). Lateral cases
lose support before slip; reverse/80 N cases cross the CoM band before support
loss. This separates investigation paths without changing the rejected gate.

Use the retained lateral-slip and reverse-push trajectories to locate the first
support/slip transition before large body motion. Compare measured contact forces
and accelerations with the accepted QP predictions at those states. Diagnose
contact unloading, reference conflicts and unmodeled motion before selecting
one correction. Keep gains and success criteria fixed; no blind CoM gain sweep.

The hand-panel task remains a separate next milestone after a defensible support
envelope is established. Runtime profiling is also separate from this physical
diagnosis. Reproduce in a fresh directory:

```bash
PYTHONPATH=src:experiments python experiments/reaching_pose_validation.py --output results/staging/pose-validation-new
```
