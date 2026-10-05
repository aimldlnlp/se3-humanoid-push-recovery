# Retained Numerical Hierarchy Follow-Ups

These are rejected numerical experiments, not portfolio baseline improvements.
Each uses the same 40 frozen states. None passes the all-state rollout gate;
all have zero physical trials and no production promotion.

| Implementation | Accepted states | Retained study |
|---|---:|---|
| Equality and task null spaces, OSQP | 10/40 | [Study](../hierarchy_study_nullspace/study.json) |
| Reduced SciPy quadratic solve | 29/40 | [Study](../hierarchy_study_nullspace_scipy/study.json) |
| Separate constant projected rows | 30/40 | [Study](../hierarchy_study_nullspace_constant/study.json) |
| Optional Clarabel backend | 29/40 | [Study](../hierarchy_study_nullspace_clarabel/study.json) |
| Active-face roundoff allowance | 26/40 | [Study](../hierarchy_study_nullspace_roundoff/study.json) |

The baseline accepts all 40 states. The best candidate still rejects ten states.
No physical recovery, tracking improvement, or real-time result is established.
The later uncommitted lock-band variant is not included in this publication.

The [original report](../HIERARCHY_REPORT.md) documents the two earlier attempts.
Each follow-up directory retains its plan, frozen outcomes, decision and hash manifest.
The corrected weighted-QP portfolio milestone remains separate:
[baseline](../../../docs/PORTFOLIO_BASELINE.md).
