# Pose Target and Jacobian Convention Audit

## Decision

The production pose task mixes two origins. MuJoCo's body Jacobian maps qvel to
body-origin linear velocity and world angular velocity. The SE(3) error and
adjoint-transported gains use a spatial twist referenced to the world origin.
World-aligned axes do not make these quantities interchangeable. This is a
confirmed convention mismatch, not yet a demonstrated cause of held-target FALL.

Keep the contact model and gains frozen. No controller correction or physical
trial is included in this audit; the physical gate remains 2/3.

## Evidence

The same eight saved states from 2.996 to 3.436 s are replayed. All accepted
controls match exactly. Torso and pelvis are checked at two central-difference
steps (1e-6 and 5e-7 s): 32 measurements. Position/orientation finite differences
match the raw body Jacobian. The SE(3) pose-increment logarithm instead matches
the Jacobian after conversion to a world-origin spatial twist.

For pose translation p, body-origin velocity u and world angular velocity w:

```text
V_spatial = [u + p cross w; w]
H(p) = [[I, hat(p)], [0, I]]
J_spatial = H(p) J_geometric
bias_spatial = H(p) Jdot_geometric qvel + [u cross w; 0]
```

Jacobian derivatives from MuJoCo are also checked against central differences.
Maximum velocity errors are 3.45e-10 for geometric and 2.89e-9 for spatial
conventions. Maximum bias errors are 1.69e-10 and 2.63e-10 respectively.

Translate the world origin by [0.7, -0.4, 0.2] m without changing the motion.
The geometric Jacobian must stay unchanged, but the production pose target
changes. Supplying the converted spatial Jacobian makes the target obey the
spatial adjoint law: maximum error 1.37e-13. Existing synthetic frame tests
assume an adjoint-transformed input Jacobian, so they do not test the actual
MuJoCo adapter contract and cannot detect this mismatch.

At the final snapshot (3.436 s):

| Measurement | Torso | Pelvis |
|---|---:|---:|
| Raw vs spatial linear velocity difference (m/s) | 0.102389 | 0.095446 |
| Production linear target change under origin translation (m/s^2) | 0.756974 | 3.313229 |
| Diagnostic body-origin target vs production linear target (m/s^2) | 0.932132 | 3.751461 |

The diagnostic evaluates the existing pose law with spatial velocity, then
converts the resulting spatial acceleration to body-origin acceleration. Its
angular target is unchanged. These are diagnostic target differences, not new
commands, optimized residuals or recovery improvements.

## Acceleration Approximation

Production pose objectives use J qdd, without pose Jdot qvel. Physical geometric
acceleration is J qdd + Jdot qvel. Spatial acceleration additionally needs the
moving-origin derivative term above. The existing task describes a resolved
acceleration approximation, so omission of these terms is documented separately
from the demonstrated origin mismatch. At 3.436 s the torso geometric bias is
small (maximum component 0.000988); pelvis geometric bias is numerical zero.
This does not establish that the bias is negligible at every state or task.

Merely replacing the Jacobian with H J is not a complete correction: derivative
terms and the objective metric must remain consistent. Identity-weighted spatial
residual norms themselves depend on the origin. Preserving a body-origin metric
requires either conversion back to geometric residuals or a matching metric
transformation. A passing twist-equivariance test alone is not a QP-invariance
test. The nonlinear SE(3) tracking law also remains an approximation.

## Verification and Scope

98 tests pass, including two new real-G1 convention tests. Artifact hashes and
the original trajectory hash match. Scientific source is `ff413e4`. Production
tasks, Jacobians, controller, references, gains, contact equations and physical
criteria remain unchanged. New code only audits saved states and writes data.

[Measurements](frame_audit/frames.csv), [complete audit](frame_audit/audit.json).
The MuJoCo API defines the body-frame Jacobian and point-Jacobian derivative:
[official API reference](https://mujoco.readthedocs.io/en/3.3.3/APIreference/APIfunctions.html#mj-jacbody),
[official declarations](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h).

## Next Gate

Completed by the [body-origin correction pilot](POSE_ORIGIN_REPORT.md). The opt-in
candidate passes the frozen QP gate and all three paired simulation conditions;
the production controller remains unchanged and broader validation is pending.

Implement one audit-only corrected pose-target variant using a consistent
body-origin acceleration and metric, retaining the existing SE(3) pose law,
contact model and gains. First test origin invariance of its full pose objective
and acceleration bookkeeping. Then replay the frozen QPs and compare task
compromises. Only after that run the unchanged nominal/moving/hold physical
gates; do not promote it or retune CoM based on this audit alone.

```bash
PYTHONPATH=src:experiments python experiments/reaching_frame_audit.py --output results/staging/frame-audit-new
```
