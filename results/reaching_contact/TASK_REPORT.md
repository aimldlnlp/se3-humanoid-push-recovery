# Pre-Excursion Task and Constraint Audit

## Decision

Keep the contact-mode model frozen. The selected pre-excursion states support
a task-compromise explanation more strongly than early torque saturation.
CoM correction is instantaneously feasible in seven interpretable LP controls
even when attained reaching acceleration is preserved. Preserving attained
torso and pelvis accelerations as well prevents full CoM correction in six of
those seven states. This implicates the body-task combination, not reaching
alone, without proving which individual task or reference causes the fall.

No controller, gain, reference, physical threshold or experimental outcome is
changed. No new physical trajectory is generated. The prior physical gate
remains 2/3, with the held-target trial classified FALL.

## Scope and Method

Eight snapshots from the contact-mode held-target trajectory span 2.996 to
3.436 s. The push begins at 3.000 s; the 100 mm CoM band crosses at 3.440 s.
All selected CoM displacements are below 100 mm. Original initial references,
joint-reference histories and saved states are restored. All eight QPs succeed
and reproduce the saved controls with zero reported difference.

An audit-only subclass captures objective matrices from the actual builder,
then captures the full accepted solver vector during the existing row-budget
check. It does not replace the objective or solver. The audit labels are guarded
against a changed objective count and current builder order is checked in source.

Torso, pelvis, CoM, posture, reach, acceleration regularization, torque
regularization, nominal torque, old fixed body slack and new point slack are
reported separately. Weighted squared residuals plus numerical regularization
reconstruct the exact quadratic objective including the omitted constant;
this equality is checked at every snapshot. Raw norms across mixed task units
are not treated as directly comparable physical errors.

An inequality is reported near-active when its distance from a bound is within
the original solver's absolute plus relative tolerance. Equalities are listed
separately. Ray coefficient zero does not by itself prove the physical friction
or support limit is binding.

## Measured Objective Compromise

At 3.436 s, CoM displacement is 99.936 mm, still below the band. CoM task
acceleration residual norm is 4.709 m/s^2. Its weighted squared cost is 44.351
out of total 28,318.290, only 0.1566% of the objective value.

| Task | Weighted cost share at 3.436 s |
|---|---:|
| Torso | 60.07% |
| Pelvis | 6.18% |
| Posture | 20.97% |
| Nominal torque | 12.35% |
| Reaching | 0.19% |
| CoM | 0.16% |

Shares describe the implemented objective at its solution. A large cost share
does not prove causal importance or prescribe a gain ratio. Gradients, units,
reference conflicts and constraints matter. CoM residual increases from
0.960 m/s^2 before push to 4.709 m/s^2 before its position-band crossing.

No torque bound, acceleration bound, normal-force inequality or world-friction
inequality is near-active in the eight selected QPs. Four left-foot friction-ray
coefficients are near zero at 3.080 s. Dynamics, wrench synthesis and point
kinematic equalities remain enforced and can still restrict motion; absence of
saturated inequality rows is not absence of constraints.

[Objective CSV](task_audit/objectives.csv), [all rows and residuals](task_audit/audit.json).

## Instantaneous Feasibility Controls

Three LPs per state minimize CoM L1 acceleration-target deviation using the same
QP constraints. Point-slack values are fixed to their original accepted values.
Other task objectives are removed from this diagnostic, not from the controller.
Controls preserve either no other task acceleration, attained reaching
acceleration, or attained reaching plus torso and pelvis accelerations. Attained
accelerations are preserved rather than replacing them with desired targets.

Seven states yield accepted LP solutions. With reaching preserved, CoM residual
is below 3e-13 m/s^2. At 3.436 s, preserving torso and pelvis as well gives a
minimum L1 residual of 2.095132 m/s^2 instead of zero. At 3.152 s the corresponding
minimum is 1.211238 m/s^2. These are instantaneous conflicts for the combined
attained body-task outputs, not proof of a stable alternative controller.

All LP solutions choose a generalized acceleration at the existing 150 rad/s^2
or m/s^2 component limit. Their extreme solutions are not rollout candidates.
This illustrates why pointwise feasibility is not sufficient for recovery.

All three exact fixed-slack LPs at 3.080 s report infeasible. The replayed QP
still passes its unchanged numerical budget. Fixing numerical slack values
exactly and enforcing LP equalities is stricter than accepting the original QP
within tolerance. This snapshot is not used to claim physical infeasibility or
to assign a task culprit. The audit retains it, without silently relaxing the
LP or any physical success threshold. Overall LP count: 21 solved, three
infeasible diagnostic controls.

## Verification

96 local tests pass. The new test checks objective reconstruction, constraint
row accounting and validated LP controls. All trajectory and output-manifest
hashes match. Scientific audit source is `ccc703d`; prototype physics remains
the earlier `8bca4ec` trajectory. No source baseline or physical outcome is
regenerated. README, library controller, contact-mode equations, branch main,
task weights and physical thresholds remain unchanged.

## Next Gate

Completed by the [frozen pose-component audit](POSE_SPLIT_REPORT.md). It splits
torso/pelvis output rows and adds minimum-acceleration witnesses without changing
the original trajectory or this historical audit's results.

Separate torso and pelvis conflicts in frozen controls, then split each pose
task's translational and rotational components. Determine which attained
component blocks CoM restoration without demanding extreme acceleration. Only
then test one reference or priority correction, keeping this contact model
fixed. Do not jump to a CoM gain sweep from the objective shares. Physical
nominal/moving/hold gates must be rerun for any later controller change.

Reproduce with a fresh directory:

```bash
PYTHONPATH=src:experiments python experiments/reaching_task_audit.py --output results/staging/task-audit-new
```
