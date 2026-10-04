# Hierarchical QP Numerical Gate

## Decision

Do not promote this sequential-QP prototype or run physical rollouts with it.
The first implementation accepts zero of 40 frozen states. A conditioning
repair accepts only six of the same 40 states, while the pose-origin baseline
accepts all 40 and reproduces the saved controls in both comparisons.

This is a numerical implementation failure, not evidence that hierarchical
whole-body control cannot improve recovery. No physical recovery improvement
has been demonstrated. The existing disturbance result remains four of eight
additional conditions passing. Panel, payload, stepping and production
promotion remain deferred.

## Candidate

The opt-in `HierarchicalController` extends the experimental pose-origin and
material-point contact controller. It solves three intermediate QPs and one
final QP, with the following priority:

1. Minimize the existing weighted material-point contact slack.
2. Minimize weighted torso, pelvis and CoM task residuals together.
3. Minimize the reaching task residual.
4. Minimize the remaining baseline posture, nominal torque, acceleration and
   torque regularization costs.

Within the balance group the original weights still define a compromise.
Contact slack stays soft; giving it the highest task priority does not convert
all sticking equations into hard physical contact guarantees. The prototype
tests this particular grouping, not all possible hierarchical formulations.

Higher-level attained outputs are locked, not their desired targets, so an
unreachable task does not intentionally become an infeasible desired equality.
Locks use a fixed band `1e-5 * (1 + abs(attained output))`. Solver row-budget
validation remains necessary and can permit additional numerical error; this
is approximate lexicographic control, not an exact symbolic hierarchy.

All original physical constraint matrix rows and bounds remain identical.
No gain, force cone, robot model, target, external-force oracle, success
threshold or load-release rule changes. The normal library controller and
default reaching path remain unchanged. CLI activation requires all three
flags: `--contact-mode --pose-origin --hierarchical`.

## Numerical Repair and Results

The first version normalizes individual lock rows but retains redundant rows
and the original objective scale. Its source is `fb7a864`; all 40 states fail.
The failed run is retained in `hierarchy_study/`.

The second version, source `83f486f`, applies positive scalar cost normalization,
warm-starts intermediate levels, and uses the independent orthonormal row space
of each task matrix for locking. These changes preserve the intended exact
optimization priorities, but finite-tolerance solver behavior can change.
No lock-band, gain or physical-threshold search is performed.

| Saved case | Accepted candidate states | Total states |
|---|---:|---:|
| Forward passing control | 1 | 6 |
| Diagonal 45-degree passing case | 1 | 6 |
| Lateral 90-degree failure | 1 | 7 |
| Lateral 270-degree failure | 1 | 7 |
| Reverse 180-degree failure | 1 | 7 |
| 80 N failure | 1 | 7 |
| Total | 6 | 40 |

Of 34 rejections, 17 reach the iteration limit, 14 intermediate levels report
`solved` but fail the original unscaled constraint budget, two report primal
infeasibility, and one final solve fails that budget with `solved inaccurate`.
Solver status alone is therefore not counted as acceptance. Rejected solves
remain explicitly unsuccessful; fallback PD is not counted as hierarchical
success. The normalized run is retained in `hierarchy_study_normalized/`.

Local timing includes accepted and rejected frozen calls, including failed
iterations and fallback paths. It is not a closed-loop real-time benchmark:

| Controller | Median | p95 | Maximum |
|---|---:|---:|---:|
| Pose-origin baseline | 9.990 ms | 15.165 ms | 15.293 ms |
| Normalized hierarchy | 187.992 ms | 281.465 ms | 318.275 ms |

Both exceed the 4 ms control interval. Increasing iteration limits would add
cost and would not by itself establish physical benefit or real-time capability.

## Verification and Rollout Gate

The two studies reuse exactly the 40 preselected states from the failure-onset
audit. Original trajectory hashes are checked. Both baseline runs reproduce
saved torque arrays within the existing tolerance. Candidate constraint prefix
checks verify original physical rows and bounds remain untouched.

107 tests pass after the conditioning repair. Tests cover a conflicting-task
example that preserves the attained higher-priority output, and an initial G1
integration solve that retains original physical constraints. Passing these
tests does not supersede failure at disturbed frozen states.

The plan predeclares nominal, moving-push, and all nine prior disturbance cases
for physical comparison only after every frozen solve is accepted. This gate
fails, so physical trials run: **zero**. No nominal, slip or fall outcome from
this candidate is claimed, and no video is generated.

The next implementation work, if pursued, is a numerically robust hierarchy
formulation with validated output preservation and original physical bounds.
It is not a gain sweep or a physical robustness claim. Keep the existing
pose-origin baseline available; do not enable this prototype by default.

## Reproduction

Use fresh output directories; never overwrite the retained failed studies.

```bash
PYTHONPATH=src:experiments python experiments/reaching_hierarchy_study.py --output results/staging/hierarchy-new
```

[First study](hierarchy_study/study.json),
[normalized study](hierarchy_study_normalized/study.json),
[all normalized frozen results](hierarchy_study_normalized/frozen.json).
