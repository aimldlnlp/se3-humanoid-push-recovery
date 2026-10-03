# Demo Media

The public README uses two short demonstrations rather than a gallery of static plots.
Both videos are MP4/H.264, 1920 x 1080, 30 FPS, without audio. They remain below
10 MB each. The reaching video uses fast-start metadata for streaming.

## Standing Push Recovery

[Watch the comparison](../assets/demos/push-recovery.mp4).

This is the existing verified three-controller recording, copied without changing
its frames or timing. It shows pure PD, PD with nominal feedforward, and SE(3)
whole-body control under the same initial state and canonical 70 N push.
Pure PD is a feedforward-removal ablation, not a qualified standing baseline.

The existing GitHub attachment is retained as the README's inline player.

## Reaching With Telemetry

[Watch the reaching montage](../assets/demos/reach-and-balance.mp4).

Three original five-second no-push trials appear in order: front 5 cm, front
10 cm, and front 12.5 cm. The robot scene is re-rendered from saved joint states;
no dynamics are rerun. The graphs and robot use the same nearest-timestamp
samples at 30 FPS. No time stretching or interpolated physics is introduced.

- Target error is distance to the final goal, not the moving reference.
- Torso error is the measured orientation error in degrees.
- Torque utilization is the original per-actuator normalized maximum.
- Shading marks the final 0.5 s hold window. The grey trace shows the full
  recorded trial; the colored trace and cursor show playback progress.
- The third condition fails tracking while retaining balance. It is not a fall.

The README uses a linked preview for this new video until a GitHub attachment
is available. The repository MP4 is playable/downloadable through its file page;
it is not advertised as an inline attachment player.

## Regenerate

From the repository root, with the declared dependencies and FFmpeg installed:

```bash
python scripts/render_portfolio_reaching.py --output results/staging/portfolio-media
```

The output directory must be fresh. The renderer records source trajectory
hashes, source revisions and frame-to-sample indices in its manifest. Raw study
data and existing plots are retained unchanged in the experiment archive.

## Publishing

Upload `reach-and-balance.mp4` through GitHub's Markdown attachment control and
replace the linked README preview with the resulting attachment URL for an
inline player. Do not substitute a guessed URL or assume a repository file link
is an attachment. The linked preview is a functional fallback.

[Technical notes and original evidence](EXPERIMENTS.md).
