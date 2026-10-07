# Reactive reaching pilot

The frozen pose-origin controller completes a 16-second target sequence, including a target change during motion. The paired hero trial also passes with a 50 N forward torso push for 0.15 s at the retargeting event. This is a small deterministic simulation pilot, not a general workspace or disturbance-robustness result.

## Sequence

Targets are world-frame offsets from the initial right-wrist attached point. Each new segment lasts two seconds and starts from the preceding reference's position, velocity, and acceleration at the event time.

| Event time | Goal offset (cm) | Required hold window |
|---|---|---|
| 0.5 s | A: (8, 0, 0) | 3.5–4.0 s |
| 4.0 s | B: (8, -4, 2) | 6.5–7.0 s |
| 7.0 s | C: (10, -2, 4) | 9.5–10.0 s |
| 10.0 s | Return toward A | Cancelled at 10.8 s; no completion requirement |
| 10.8 s | Redirect toward B | 15.5–16.0 s |

The hero push begins at 10.8 s in world +x. Its measured impulse is 7.5 N·s. The nominal trial uses the identical schedule, reference arrays, and initial state, with no push.

## Results

| Measure | Nominal | Hero |
|---|---:|---:|
| Whole-sequence result | PASS | PASS |
| A maximum hold error | 2.083 mm | 2.083 mm |
| B maximum hold error | 8.596 mm | 8.596 mm |
| C maximum hold error | 10.230 mm | 10.230 mm |
| Final B maximum hold error | 3.675 mm | 1.920 mm |
| Moving-reference tracking RMSE | 7.694 mm | 7.396 mm |
| Maximum tracking error | 10.917 mm | 10.917 mm |
| Maximum horizontal CoM displacement | 11.242 mm | 22.478 mm |
| Maximum torso orientation error | 0.01976 rad | 0.04413 rad |
| Rejected QPs | 0 | 0 |
| Deadline misses at 4 ms | 100% | 100% |

Every required hold meets the unchanged 15 mm tolerance. Moving RMSE also meets the predeclared 15 mm gate. Final balance, actuator, numerical, and whole-trial physical checks pass. The classifier observation horizon covers the full 16-second trial; physical thresholds are unchanged. Its whole-trial `physical.recovered_at_s` and `recovery_latency_s` fields describe initial settling from startup, not recovery after the hero push; no disturbance recovery-latency claim is made.

## Verification and implementation

- The original corrected 70 N hold trial passes and reproduces saved baseline states, controls, hand tracking, and measured contact wrenches within the existing replay tolerance.
- The existing baseline suite passes 106 tests, excluding the unfinished optional hierarchy suite. Two new tests pass: reference continuity and endpoints, and assessment of a late fall with an intentionally cancelled goal.
- The controller, gains, contact model, weights, and physical constraints are reused. Only the reference generator, sequence assessment, and renderer are new. The library default and existing experimental files are unchanged.
- Source hashes in `plan.json` match the executed source. The base commit is `4ee50e2`; this is a working-tree snapshot with uncommitted additions, not a newly committed checkpoint.
- Execution uses local Python 3.12.0, NumPy 1.26.3, SciPy 1.12.0, MuJoCo 3.14.0, and OSQP 1.1.3. SSH source/trajectory upload was rejected by automatic approval review; no snapshot was uploaded. A separate matching worker environment was prepared but did not execute this pilot.

## Media and reproduction

[Hero progression video](hero/media/reactive-reaching.mp4) and [nominal control video](nominal/media/reactive-reaching.mp4) replay saved states at the original simulation speed: 480 frames, 30 fps, 1280×900, H.264. Goal markers, short reference/hand trails, push events, and measured foot loads are shown. Frames during the push and at final hold were visually checked. Motion and body response remain conservative; PASS does not establish dramatic whole-body motion or arbitrary-target capability.

From the repository root, in the pinned portfolio environment:

```bash
python scripts/run_reactive_reaching.py --output results/staging/my-reactive-pilot
python -m pytest -q tests/test_reactive_reaching.py
```

Use a fresh output directory. The runner first checks the old held-target regression, then runs nominal, and runs hero only if nominal passes. It retains attempted trials and does not change gains in response to outcomes. Raw data, the initial state, the predeclared plan, verification, and a byte-hash manifest accompany this report.
