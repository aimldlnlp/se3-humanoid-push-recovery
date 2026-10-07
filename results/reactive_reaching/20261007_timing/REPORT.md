# Reactive reaching: push timing boundary

The three-time validation gate fails: two cases pass, and the earlier push causes a fall. Publish the successful sequence with this timing limit visible. Do not promote the controller or claim general reactive recovery.

## Frozen experiment

The 16-second A/B/C sequence and the redirection at 10.80 s are unchanged from the [original pilot](../20261006_pilot/REPORT.md). All three trials use its saved initial state and identical position, velocity, acceleration, and goal reference arrays. The only experimental change is push start time. Every push is 50 N in world +x for 0.15 s, with a measured impulse of 7.5 N·s. Gains, weights, contact equations, and physical thresholds are unchanged.

The case list was recorded before outcomes. All cases completed, including the failed case; there was no tuning after the first failure. The separate nominal pilot remains the no-push reference.

## Measured results

| Push start | Relative to redirection | Result | Moving tracking RMSE | Final B hold maximum error | Rejected QPs |
|---|---|---|---:|---:|---:|
| 10.55 s | 0.25 s before | FALL | 27.881 mm | 329.112 mm | 0 |
| 10.80 s | Simultaneous | PASS | 7.396 mm | 1.925 mm | 0 |
| 11.05 s | 0.25 s after | PASS | 7.498 mm | 2.460 mm | 0 |

Passing cases meet every mandatory waypoint's 15 mm hold tolerance, the 15 mm moving-reference RMSE gate, final balance, and whole-trial physical checks. The cancelled A goal has no completion requirement. The early case's final error is measured after the fall; it is not a diagnosis of what initiates the failure. Valid QP solutions do not guarantee physical recovery.

These are three sampled timings, not a continuous safe interval or a probabilistic success estimate. All runs miss the 4 ms control deadline. The robot remains a fixed-foot simulation with virtual position goals; no hardware, grasping, walking, or general robustness claim is made.

## Media

[Watch or download the successful 16-second progression](simultaneous/media/reactive-reaching.mp4).

The video replays the simultaneous case, with A/B/C goal labels, a hand close-up, short reference/hand trails, the actual push interval, and measured foot loads. It uses 480 frames at 30 fps, 1280×900 H.264, and original simulation speed. It does not combine successful fragments from different trials. Source states, raw contact data, and failures remain available for all three cases.

[Early failed trial summary](before/summary.json) · [All outcomes](study.csv) · [Verification](verification.json)

## Verification

- 108 tests pass, with `tests/test_reaching_hierarchy.py` intentionally excluded as an unfinished optional experiment.
- Initial positions and velocities, all reference arrays, and time arrays are paired exactly against the saved nominal. Measured force traces and 7.5 N·s impulses match the declared push conditions.
- Active controller dependencies match the pilot source. The study executed before the publication commit; its plan retains the original runner hash. Later edits made dependency checks portable across checkout line endings and added a render-only output option without changing the executed physical trial or rendering functions. Final source hashes are recorded in `verification.json`.
- Physics ran locally with Python 3.12, NumPy 1.26.3, SciPy 1.12.0, MuJoCo 3.14.0, and OSQP 1.1.3.
- The legacy `physical.recovered_at_s` and `physical.recovery_latency_s` fields refer to initial settling from startup in this whole-trial assessment. They do not measure recovery after the push, and can exist even for a trial that later falls.

## Reproduce

Fetch Git LFS artifacts and use the pinned portfolio environment. From the repository root:

```bash
git lfs pull
python -m pip install -r requirements-portfolio.txt
python scripts/run_reactive_reaching.py --output results/staging/my-timing-study --previous results/reactive_reaching/20261006_pilot
```

Use a fresh output directory. The timing runner retains every outcome and renders the simultaneous case after completing all three trials. To render saved states into a separate new directory without rerunning physics:

```bash
python scripts/run_reactive_reaching.py --output results/reactive_reaching/20261007_timing/simultaneous --render-only --media-output results/staging/my-reactive-media
```
