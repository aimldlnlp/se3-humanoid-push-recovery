# G1 revalidation artifact — source `6024ce9`

This directory contains the final fixed-foot Unitree G1 revalidation generated from source checkpoint `6024ce9af3b63d62c584d120fe2309ef10297198`. It is intentionally separate from the previous baseline under `results/data/`, so old measurements are not overwritten or relabeled.

## Runs

| Run | Contents | Trials |
|---|---|---:|
| `final-6024ce9-g1` | standing, perturbed standing, canonical 70 N push | gate runs |
| `final-6024ce9-push-calibration` | 10, 20, 40, 60, 80, 100 N × 8 directions × PD/WBC | 96 |
| `final-6024ce9-push-sweep` | 10–80 N × 24 directions × PD/WBC | 384 |
| `final-6024ce9-robustness` | friction, mass-scale, and push-duration perturbations over five seeds | 50 |

All manifests carry the same configuration hash:

`6bd9912b7daf2a3534264cb2ca327b04151773dd5ac33293be02475b90f53f72`

## Measured outcomes

- Canonical 70 N push: PD `FALL`; SE(3) WBC recovered.
- Push sweep: PD `32/192`; SE(3) WBC `145/192`.
- Robustness: `32/50` SE(3) WBC trials recovered.
- Calibration: at 80 N, PD `0/8` and SE(3) WBC `2/8`; at 100 N, both `0/8`.
- Canonical SE(3) WBC: peak torso error `0.0507 rad`, peak horizontal CoM displacement `0.0918 m`, maximum joint torque `20.28 N m`, recovery latency `0.270 s`.
- Canonical QP/physical-contact comparison: total-force RMSE `7.63 N`; vertical-GRF RMSE `13.18 N`.

## Directory map

- `data/` — raw NPZ trajectories and CSV trial tables.
- `figures/` — selected PNG, vector PDF, and SVG outputs regenerated from this run.
- `videos/` — G1 hero, synchronized comparison, and perturbed-standing H.264/GIF outputs.
- `logs/` — run manifests and render provenance.

Execution provenance for this artifact is recorded in the manifests. The package makes no hardware-validation claim, and its numerical results should be interpreted only with the recorded source, configuration, model, and environment metadata.
