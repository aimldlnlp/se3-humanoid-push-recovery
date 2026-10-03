# Measured-balance guard pilot

## Decision

Neither tested policy qualifies. Keep reach weight 100 and all controller
defaults unchanged. All four guarded pushed trials fall; validation is gated
off for both prototypes. Retain these negative results for reproduction.

## Frozen protocol

- Front target: 0.125 m along world X, same wrist point and initial state.
- Five-second window, seed 0, existing 15 mm final-hold tracking tolerance and
  original whole-task physical criteria; no threshold relaxation.
- Paired fixed-high weight 3000 versus guard, with no push, moving push at
  1.5 s and hold push at 3.0 s; both pushes 70 N for 0.15 s (10.5 Ns).
- Guard reduces immediately to 100 if torso orientation exceeds
  0.0872664626 rad, angular speed exceeds 0.15 rad/s, CoM XY displacement
  exceeds 0.10 m, or either measured contact flag is absent.
- After 0.25 s continuously inside all bands, restore linearly over 0.25 s;
  renewed risk immediately reduces authority. No pulse timing/magnitude input.
- Arm-adaptation variant updates only seven right shoulder/elbow/wrist posture
  references to measured positions while reduced, including the fallback
  reference. Damping, gravity, objective weights and other joint references
  remain unchanged; restored high authority retains the last arm reference.
- Right 5 cm, up 5 cm and mixed 8 cm validation was prescribed only after
  all three guarded pilot conditions pass. Neither gate passes.

## Results

| Policy | Condition | Outcome | Hold max error, mm | First reduction, s | First post-step contact loss, s | QP failures |
|---|---|---|---:|---:|---:|---:|
| Fixed high, both pilots | No push | PASS | 3.231 | none | none | 0 |
| Weight guard | No push | PASS | 3.231 | none | none | 0 |
| Arm-adaptation guard | No push | PASS | 3.231 | none | none | 0 |
| Fixed high, both pilots | Moving | FALL | 866.904 | none | 2.908 | 0 |
| Weight guard | Moving | FALL | 1696.622 | 1.520 | 3.244 | 1 |
| Arm-adaptation guard | Moving | FALL | 1468.665 | 1.520 | 4.104 | 1 |
| Fixed high, both pilots | Hold | FALL | 722.759 | none | 3.824 | 1 |
| Weight guard | Hold | FALL | 1634.966 | 3.020 | 3.856 | 2 |
| Arm-adaptation guard | Hold | FALL | 1278.807 | 3.020 | 3.864 | 2 |

The table combines identical fixed-high results for readability; both pilots
executed all six trials, totaling twelve fresh physical runs. Reduction remains
active for the rest of both pushed trials: 3.48 s moving and 1.98 s hold.
Delayed post-step contact loss is not success: both balance and tracking fail.
Guard decisions use pre-step contact flags; event times above use the distinct
post-step contact channels, consistent with physical outcome assessment.

## Provenance and verification

Physics ran on `hucenrotia-ai`, Linux CPU, Python 3.12.3, MuJoCo 3.11.0,
OSQP 1.1.3 and NumPy 2.5.2. Source snapshots:

- Weight-only: `07ffc0fa2283bee03d59f643d48c2a167a45f7fd`.
- Arm-adaptation: `11ebc63ab6c4d23f64b99bb70f653d21d9982f49`.

Within each pilot, actual initial states, prescribed targets and force inputs
match across pairs; the no-push guarded and fixed trajectories are identical.
Historical Windows baseline replay passes with explicit floating-point
tolerances; byte-level initial-state hashes are not claimed equal cross-host.
The two worker pilots also reproduce the same fixed-high dynamics.

The verifier recomputes all twelve outcomes and replays all six guarded weight
sequences from measured state. For arm adaptation it also checks that non-arm
references remain constant, reduced arm references equal measured positions,
and references remain held when high authority returns. Artifact hashes cover
raw trajectories, summaries, logs, original plots and separate presentation
plots. Both presentation plots were visually inspected; no dynamics rerun was
used to generate them. No video is included in this diagnostic batch.

Full suites pass 83 tests locally and on the worker at the arm-adaptation source.
Timing is shared-worker CPU timing, not hard-real-time evidence. Some summary
residuals contain NaN where QP fallback occurs; trajectories used for outcome
assessment are finite. QP failure counts remain visible, not filtered away.

[Weight-only data](study.csv), [verification](verification.json),
[manifest](manifest.json), [presentation plot](presentation/guard_results.png).
[Arm-adaptation data](../11ebc63_arm_worker/study.csv),
[verification](../11ebc63_arm_worker/verification.json),
[manifest](../11ebc63_arm_worker/manifest.json),
[presentation plot](../11ebc63_arm_worker/presentation/guard_results.png).

## Next diagnostic

Compare QP-predicted contact acceleration, slack and force with measured foot
velocity, displacement and contact state after the pulse but before contact
loss. A fixed-contact acceleration model may not arrest accumulated foot drift;
that remains an unverified hypothesis. These pilots do not establish a unique
root cause and do not justify another blind gain sweep or default promotion.
