# Contact physics, runtime, and model-mismatch pilot

Simulation source: `eaef962e7cc24cee0cf499047803244eada5fb2f`.
The source archive SHA-256 is
`9035bb5930f03ef1e7792be3c28b94eac9ed9586575e9e79a1d6dc2e53c82530`.
The checksum matched before remote extraction. Tests passed locally and
on the execution worker: **65 passed** on each host. The worker used
MuJoCo 3.11.0 and OSQP 1.1.3. Existing OSQP deprecation warnings remain.

## What changed

The old component-wise box admitted diagonal forces outside the circular
Coulomb bound. Production QP rows now impose the conservative inner diamond
$|F_x|+|F_y|\leq\mu F_z$, with unchanged CoP/torsion rows, gains, actuator
limits, and recovery thresholds. This is a world-XY approximation for the
horizontal-ground experiment, not an exact distributed foot wrench cone.
It does not reproduce MuJoCo's soft-contact law or its full coupled
torsional/rolling contact cone.

OSQP workspace reuse updates values only for identical CSC patterns,
contact ordering, dimensions, and solver settings. Otherwise it rebuilds.
Failed solves discard the cached workspace. Reuse remains **opt-in**, via
`controller['solver']['reuse_workspace'] = True`.

Explicit WBC model mismatch uses a separate internal model. Nominal torque
and fallback gravity terms use that model rather than oracle plant parameters;
returned commands still obey the physical actuator limits.

## Protocol

18 trials ran from one frozen source archive. The nominal snapshot is settled
once for 0.6 s. Cold/warm pairs share the snapshot and exact physics-substep
force trace. Canonical comparison includes Pure PD, PD + nominal FF, cold WBC,
and warm WBC. The 70 N, 0-degree, 0.15 s pulse realizes 10.5 N s impulse.
Push trials observe 6.1 s after pulse end; the classifier retains its
six-second recovery deadline.

Eight additional trials compare known versus unknown plant parameters on
the same physical plant. Factors are mass scale 1.1 and plant friction 0.4.
Unknown-model WBC retains controller mass scale 1.0 and friction 0.7.
Seeds 0 and 1 add independent base angular-rate perturbations (Gaussian
standard deviation 0.01 rad/s), push-magnitude jitter uniform in
[0.92, 1.08] times 70 N, and direction jitter Gaussian with standard deviation
8 degrees. Paired knowledge conditions share state and force hashes.

## Measured outcomes

| Condition | Cold WBC | Warm WBC |
|---|---|---|
| Quiet standing | Recovered | Recovered |
| 70 N at 0 degrees | Recovered, 0.270 s latency | Recovered, 0.270 s latency |
| 40 N at 90 degrees | Recovered | Recovered |
| 70 N at 90 degrees | SLIP | SLIP |

Both PD variants fall in the canonical comparison. Cold canonical WBC has
peak torso error 0.0505658 rad, peak horizontal CoM displacement 0.0811918 m,
and peak torque 20.2749 N m. These match the old canonical result to reported
precision; this does not establish unchanged results across the full sweep.

With the 1.1 mass-scale plant, both seeds recover under known and unknown
models. Latency changes from 0.218/0.254 s to 0.282/0.302 s when the controller
retains nominal mass. Relative mass-matrix mismatch is approximately 0.0909.
With friction 0.4, seed 0 recovers and seed 1 slips under both knowledge
conditions. Those trajectories coincide in this pilot; the changed QP friction
bound is inactive. Two seeds cannot establish broad robustness or a population
success rate.

## Runtime

Times include the entire production `solve()` call, not just OSQP iterations.
Build, prepare/setup/update, and numerical solve times are separately available
in [timing.csv](timing.csv). One sequential run per condition was measured
on a shared worker, with BLAS/OpenMP thread counts set to one. No CPU affinity
or hard-real-time scheduling was used; this is not a replicated speedup claim.

| Canonical WBC | Mean [ms] | p95 [ms] | p99 [ms] | Max [ms] | >4 ms [%] |
|---|---:|---:|---:|---:|---:|
| Cold | 3.872 | 6.454 | 6.829 | 8.787 | 28.260 |
| Reused | 2.852 | 5.286 | 5.596 | 15.091 | 12.845 |

Reuse lowers canonical mean time by approximately 26.3%, but not every tail
metric improves. Quiet-standing deadline misses increase from 13.13% to
19.93%, and lateral 70 N peak torque changes from 20.979 to 23.453 N m.
Classification remains SLIP in both lateral trials. Reuse is not promoted
to the default, and neither configuration meets a hard 4 ms deadline.

## Validation and numerical limitations

All 18 NPZ files open without pickle. State, control, actual GRF, source
commit, initial-condition hashes, force traces, and recovery metadata were
checked. Timing summaries were recomputed from 31,883 call records.
All WBC calls succeed. PD bootstrap calls are also included in the timing
archive but excluded from WBC timing summaries.

The strict 0.001 N inner-diamond residual check **failed** on one lateral cold
trial: maximum positive L1 excess is 0.0100845 N. It is reported, not rounded
to zero. Solver tolerances remain unchanged (`eps_abs = eps_rel = 0.001`,
scaled termination enabled). The physical circular Coulomb excess is at most
3.55e-15 N. Maximum dynamics and contact-equation residual norms are
0.0007014 and 0.0000747, respectively. Contact equations include the explicit
penalized slack; small residual does not prove zero physical foot acceleration.

The [summary](summary.json), [trial CSV](pilot.csv), and
[validation receipt](validation.json) are committed. Full trajectories and
test logs remain in the isolated execution archive and local ignored staging.
The old 192-condition sweep, figures, and videos are unchanged and remain
explicitly labeled with their old source checkpoint. This pilot is not their
replacement; rerun the complete paired benchmark before promoting a new basin.

## Reproduction

```bash
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/contact_audit_pilot.py --output results/staging/contact-audit-new-run
```

Use a fresh output directory; the script refuses to overwrite an existing one.
