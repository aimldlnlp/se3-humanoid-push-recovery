# Contact Wrench Audit and Current-Patch Pilot

## Decision

The QP can request wrenches inconsistent with the reconstructed instantaneous
contact geometry. This is a supported model mismatch, not a unique explanation
for failure: mismatches also appear in the successful damping-only moving trial.
One current-patch moment-bound candidate is tested and rejected. Defaults and
success thresholds remain unchanged; multi-target validation remains blocked.

## Method

Eighteen frozen states from four saved trajectories are replayed: damping-only
hold/moving and pose-restoration hold/moving. Actuator commands match saved
commands with zero reported difference on this runtime. Each foot is checked
against an outer friction cone generated at all reconstructed geometric ground
contacts. Unilateral normal forces, contact-frame tangential coefficients,
available spin and rolling moments, and moments about the foot-body origin are
included. No force-bearing contact history is fabricated.

A linear program minimizes the L1 wrench residual. Force components use unit
scale and moment components use a fixed 0.1 m scale; the result is normalized
by the correspondingly scaled target L1 norm. Outer box friction bounds enlarge
the true cone. Acceptance does not prove feasibility; rejection means that even
this enlarged instantaneous geometric cone cannot reproduce the wrench.
Contacts that may be unloaded are included, making rejection conservative.

The original NNLS solver encountered redundant, rank-deficient generators and
stopped without producing a completed audit. The LP replacement completes;
only its completed bundle is published. The reconstructed geometry remains
distinct from the original MuJoCo force solver's state and contact history.

## Findings

- 25 of 36 sampled foot predictions have normalized residual above `1e-6`.
- All 36 original measured pre-step wrenches have reported residual zero.
- Damping hold at 4.284 s, before first slip-speed crossing: left/right predicted
  residuals 0.101465 / 0.057163, with one geometric contact per foot.
- Damping moving at 4.500 s, despite eventual PASS: residuals 0.039892 / 0.089271.
- Both before-push flat-foot snapshots and damping moving at 4.900 s fit the
  enlarged cone. Geometry changes matter, but mismatch alone does not classify
  recovery or establish causality.

CoM force balance is also inspected. At 4.900 s in the pose-restoration moving
TIMEOUT, forward CoM displacement is 106.112 mm and forward velocity is
-0.032386 m/s. The CoM task requests -3.390061 m/s^2 forward acceleration, while
the QP's net contact forces imply -0.177671 m/s^2. Measured forces imply
-0.116395 m/s^2 and next-interval velocity implies -0.116622 m/s^2.

Thus the QP itself does not realize most of the requested CoM correction;
plant-versus-QP acceleration mismatch is not the whole explanation. This is
consistent with competing objectives and kinematic/dynamic constraints, but
does not identify one objective weight as the cause. Net-force acceleration
includes gravity and measured pulse for the measured signal. Other non-foot
contacts are excluded; selected snapshots precede falling/other-body contact.
Instantaneous force acceleration and next-interval averages are distinct.

[Wrench CSV](wrench_audit_lp/wrench.csv), [snapshots and CoM data](wrench_audit_lp/audit.json).

## One Candidate

Starting from damping 20 s^-1, the opt-in experiment replaces the four static
CoP moment rows per foot with bounds derived from current geometric contact XY
extents relative to the body origin. The rows include the moment contribution
of tangential force at contact height. A conservative allowance covers contact
height spread and available spin/rolling moments. With no geometric contacts,
normal force is constrained to zero. No measured force enters the controller.

This is only a bounding-box approximation. It does not impose the complete
contact cone, change the six-dimensional body-contact acceleration constraint,
or implement an edge/point contact state machine. Yaw/friction constraints and
all objectives remain unchanged. The candidate does not use pose restoration.

| Front 12.5 cm, reach weight 3000 | Damping baseline | Current-patch bounds | Hold max error | QP failures |
|---|---|---|---:|---:|
| No push | PASS | PASS | 3.234 mm | 0 |
| Moving push, 70 N at 1.5 s | PASS | FALL | 702.618 mm | 0 |
| Hold push, 70 N at 3.0 s | FALL | FALL | 714.796 mm | 0 |

The three-condition gate fails at 1/3. Nominal qpos, qvel and control arrays are
exactly equal to the damping baseline; pushed conditions degrade or remain
failed. Solved QPs do not mean physical recovery. The candidate is not promoted
and there is no second candidate or gain sweep in this milestone.

## Verification

92 tests pass locally, including point-contact moment rejection, contact
spin/rolling capability, height-aware patch rows, and disabled-option behavior.
All three pilot outcomes are independently reclassified; input pairs match the
damping baseline. All 19 files listed in the new run manifests pass SHA-256
checks. Saved trajectories and existing result classifications are preserved.

Audit scientific source: `a7fdce2`. Pilot scientific source: `c559131`.
Physics uses local Windows CPU, not a cross-host replication. All three new
trials record 100% 4 ms deadline misses. No real-time claim. The library
controller, default contact behavior, reach weight 100 and physical outcome
thresholds are unchanged. README stays portfolio-focused and is not expanded.

## Next Gate

Follow-up completed: [matching contact-mode prototype and pilot](MODE_REPORT.md).
Frozen geometry gate passes; physical pilot still fails at 2/3. Original next
gate rationale follows for chronology.

Stop adding scalar contact corrections. First prototype a coherent contact-mode
formulation in frozen states: flat patch versus edge/point must have matching
wrench capability and kinematic constraints, rather than a smaller wrench box
combined with the same fully fixed foot-body acceleration. Check the measured
wrench as a positive control and verify feasible acceleration/control behavior
before running the three physical gates again. Separately measure objective
residuals for the moving CoM compromise before altering priorities. Neither
larger gains nor relaxed thresholds follow from this audit.

Reproduction, with fresh output directories:

```bash
PYTHONPATH=src:experiments python experiments/reaching_wrench_audit.py --output results/staging/wrench-audit-new
PYTHONPATH=src:experiments python experiments/reaching_contact_pilot.py --patch-bounds --output results/staging/patch-pilot-new
```
