# Portfolio Baseline

## Scope

The reaching milestone uses the existing `PoseOriginController` on top of
`ContactModeController`: material-point contact forces and acceleration
constraints, contact velocity damping at 20/s, and consistent body-origin pose
acceleration targets. The target is 12.5 cm forward; reach weight is 3000.
No gain or success-threshold changes are introduced by the portfolio runner.

This selects an explicit demo baseline, not a production-default promotion.
Hierarchical QP is a separate unfinished experiment. The runner never imports
or activates it, and the portfolio environment does not install Clarabel.
Historical experiments and failed candidates remain available unchanged.

## Reproduce

Use Python 3.12 from the repository root:

```bash
python -m pip install -r requirements-portfolio.txt
python -m pip install -e . --no-deps
python scripts/run_portfolio_reaching.py --output results/staging/my-reach --render
```

The output directory must not already exist. The command runs eleven five-second
simulations: three corrected baseline conditions, the predecessor hold condition,
a lateral failure control, and six initial-rate sensitivity conditions.
Rendering adds no physical trials and does not change playback speed.

Outputs include `study.csv`, `study.json`, `plan.json`, `perturbations.json`, saved
initial states, complete trajectories, logs, a hash manifest, and
`media/reach-before-after.mp4`. The initial states are injected after the nominal
setup, with no subsequent warmup that could erase the perturbation.

For a single corrected held-target trial, without the comparison or sensitivity grid:

```bash
python experiments/reach_and_balance.py --output results/staging/hold-demo --target-offset-m 0.125 0 0 --reach-weight 3000 --contact-mode --pose-origin --contact-velocity-damping 20 --push-N 70 --push-start-s 3 --push-direction-deg 0 --render
```

The portfolio runner additionally applies the whole-trial physical assessment
from `reaching_workspace.assess_trial`, not just the single trial's recovery flag.

## What Is Verified

The predecessor and corrected hold trials have paired initial states, target
references, time arrays, and force traces. The three corrected trajectories are
compared with the earlier saved pilot using the existing dynamics replay tolerance.
Criteria retain the 15 mm final-hold hand tolerance, 100 mm CoM band, and original
contact, slip, orientation, actuator, and numerical checks.

Two predeclared seeds (17 and 29) generate angular-rate offsets with a 0.05 rad/s
upper bound. Each is tested without push, with a moving push, and with a hold
push. Positions and standing references do not change. Actual offset vectors
and magnitudes are saved; the upper bound is not the magnitude of every sample.
This is a small sensitivity check, not a probabilistic robustness estimate.

Baseline regression tests:

```bash
python -m pip install pytest
python -m pytest -q --ignore=tests/test_reaching_hierarchy.py
```

The baseline suite passes 106 tests in the verified environment. The exclusion
is intentional and visible: unfinished optional hierarchy tests are not part of
this milestone. It is not a claim that the entire current working tree passes.

## Engineering Story

The fresh portfolio run reproduces all three corrected pilot trajectories within
the existing tolerance. Its six sensitivity trials also pass. Actual initial
angular-rate offsets are 0.00076448 rad/s and 0.01191059 rad/s; these are small,
not tests at the 0.05 rad/s upper bound. Their maximum final-hold hand errors
range from 3.769 to 5.711 mm. The predecessor remains FALL and the lateral control
remains SLIP. All eleven runs have zero rejected QPs and 100% deadline misses.

[Complete portfolio results](../results/portfolio_reaching/study.csv) |
[Saved plan and source hashes](../results/portfolio_reaching/plan.json) |
[Verification](../results/portfolio_reaching/verification.json)

The publication manifest hashes result/media bytes exactly. Source entries use
LF-normalized hashes so Git checkout line endings do not masquerade as code changes.
The original plan retains the raw source hashes captured before simulation.

Higher reach authority initially improved hand accuracy but caused pushed trials
to fall. Reducing reach authority with a guard did not recover balance. Contact
damping rescued moving-push recovery; consistent material-point contact modeling
reduced moving-case foot drift. Hold still failed. Correcting the pose origin
and acceleration bookkeeping then changed the paired hold outcome from FALL to
PASS, with 4.804 mm maximum final-hold hand error and 6.303 mm maximum foot drift.

The pose correction combines origin consistency and acceleration-bias compensation;
the pilot does not isolate their individual causal contributions. A later
disturbance check passes only four of eight new conditions. Successful QP solves
alone do not guarantee physical balance or no-slip contact.

## Limits

- Fixed-foot simulation on flat ground; no hardware, grasping, payload, or walking claim.
- Reaching remains experimental and fails some lateral, reverse, and higher-force cases.
- The local corrected reaching trials miss the 4 ms deadline; no real-time claim.
- The six initial-rate trials do not establish a workspace or disturbance envelope.
- Numerical hierarchy experiments have no validated physical recovery improvement.

[Pose correction evidence](../results/reaching_contact/POSE_ORIGIN_REPORT.md) |
[Disturbance limits](../results/reaching_contact/DISTURBANCE_REPORT.md) |
[Video provenance](DEMO_MEDIA.md)
