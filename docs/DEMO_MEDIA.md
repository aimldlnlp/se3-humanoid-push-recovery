# Demo Media

The public README uses two short demonstrations rather than a gallery of static plots.
Both current videos are MP4/H.264, 1920 x 1080, 30 FPS, without audio.
The reaching video uses fast-start metadata for streaming.

## Standing Push Recovery

[Watch the comparison](../assets/demos/push-recovery.mp4).

This is the existing verified three-controller recording, copied without changing
its frames or timing. It shows pure PD, PD with nominal feedforward, and SE(3)
whole-body control under the same initial state and canonical 70 N push.
Pure PD is a feedforward-removal ablation, not a qualified standing baseline.

The GitHub attachment was refreshed after the old attachment URL stopped working.
The original repository video remains unchanged. Both current attachments are
retained in the closed media archive issue linked below.

## Reaching Before and After

[Watch the paired comparison](../assets/demos/reach-before-after.mp4).

Three five-second trials run side by side. The first two have identical initial
states, 12.5 cm front targets, and 70 N forward pushes at 3 s for 0.15 s.
The predecessor already has the material-point contact correction but falls.
Adding the pose-origin and acceleration-bias correction passes. The third panel
uses the corrected controller with a lateral 90-degree push and fails with slip.

- Target distance is distance to the final goal, not the moving reference.
- CoM displacement uses the saved standing reference, not a fitted reference.
- Foot drift is the maximum measured post-step displacement across both feet.
- Foot drift displays `n/a (contact lost)` when double support is absent;
  a missing contact must not look like zero drift.
- Torque utilization is the measured per-actuator normalized maximum.
- Outcome labels describe the whole trial, not the instantaneous frame.
- The third panel illustrates a limitation, not an additional success.

The robot and telemetry use the same nearest-timestamp samples at 30 FPS.
Rendering replays saved qpos; it does not rerun dynamics, interpolate physics,
or stretch time. The comparison is not PD versus WBC and is not a hierarchy demo.

The README embeds the uploaded GitHub attachment as an inline player. The
repository MP4 remains a download fallback; its file page is not a video player.
The attachment is retained in the closed
[media archive issue](https://github.com/aimldlnlp/se3-humanoid-push-recovery/issues/1).

## Regenerate

From the repository root, with the declared dependencies and FFmpeg installed:

```bash
python scripts/run_portfolio_reaching.py --output results/staging/portfolio-run --render
```

The output directory must be fresh. This runs the physical comparison and six
initial-rate sensitivity trials before rendering the three selected panels.
To render an existing run without new simulations:

```bash
python scripts/run_portfolio_reaching.py --output results/staging/portfolio-run --render-only --media-output results/staging/portfolio-media-new
```

Render-only requires an existing trial tree and a fresh media output directory. The
renderer records trajectory hashes and frame-to-sample indices in
`media/selection.json`. Historical data and the earlier no-push reaching video
remain unchanged in the archive.

## Publishing

For a regenerated video, upload `reach-before-after.mp4` through GitHub's Markdown
attachment control or GitHub CLI's `--attach` option. Put the resulting attachment
URL in its own README paragraph for an inline player. Do not substitute a guessed
URL or assume a repository file link is an attachment.

[Technical notes and original evidence](EXPERIMENTS.md).
