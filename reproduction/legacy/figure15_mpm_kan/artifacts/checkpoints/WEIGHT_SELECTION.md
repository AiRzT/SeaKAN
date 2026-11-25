# Figure 15 weight selection

## Released checkpoint

Use `figure15.pt` for the released Figure 15 profiles and contours.

- SHA-256: `32b801b4db3b7739d5ed74b8a1a8b80ce4ab6748f5606dfcf8eb1014632fb435`
- Source: `outputs/candidate56a_wall_physics/checkpoints/figure15.pt`
- Water checkpoint SHA-256: `381f4e5a9cf89e0d5f464dd38191bc2115c6e7797fdc180fb866417fc5302859`
- Device and finish precision: NVIDIA GeForce RTX 4090 D and `torch.float64`

The combined checkpoint retains the released Test 10 soil KAN, Test 49
soil KAN and contact-enriched porous-interface KAN. Only the exposed-wall
harmonic-RANS KAN was refined.

## Physics-only refinement

The Figure 15 CSV files were not used in the loss, gradients or teacher
initialisation. Starting from the version 1.3.0 water checkpoint, 30 Adam
updates were performed with the pipe-wall momentum residual weight raised
from 0.5 to 10. The change addresses the previous loss imbalance between
the free-surface and pipe-wall equations. Validation data were loaded only
after the checkpoint had been frozen.

## Validation metrics

| Comparison | Relative L2 | RMSE | R2 |
|---|---:|---:|---:|
| Test 10 experimental points | 10.04% | 0.05159 | 0.9443 |
| Test 10 numerical pore-pressure curve | 11.10% | 0.06928 | 0.8876 |
| Test 49 experimental points | 8.29% | 0.06659 | 0.8718 |
| Test 49 numerical pore-pressure arc | 5.86% | 0.03811 | 0.9396 |
| Test 49 numerical hydrodynamic-pressure arc | 1.90% | 0.01702 | -0.3492 |
| Test 49 combined numerical interface | 3.54% | 0.02824 | 0.9710 |

The isolated hydrodynamic arc has a reference coefficient of variation of
only 1.64%. Its standard R2 is therefore dominated by a small mean offset
and is not used as the global Test 49 score. The combined interface score
uses all 206 numerical reference samples from the exposed and embedded
arcs. The negative arc value is retained here for auditability; it is not
clipped or replaced by a fitted score.

Historical local paths in the checkpoint metadata were reduced to filenames for this release. The numerical tensors are preserved.
