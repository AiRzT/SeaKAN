# Rebuilding the revision figures

Run the plotting command from the repository root. It reads the archived tables, profile arrays, COMSOL fields and weights. Use a fresh output directory when comparing versions.

```bash
python reproduction/generate_revision_figures.py --output-dir outputs/revision_figures
```

To regenerate selected panels, supply their manuscript figure numbers.

```bash
python reproduction/generate_revision_figures.py --figures 5 6 --output-dir outputs/flume_figures
```

| Figure | Inputs and operation |
|---|---|
| 1, 2 | Draw the geometry and network architecture from the plotting code. |
| 5, 6 | Combine digitized reference profiles with archived SeaKAN predictions for the buried-pipe benchmark and trenched-pipe Tests 10 and 49. |
| 7 | Read model-comparison profiles, scores and the Standard KINN result. |
| 8 | Load the validation reference field and the selected physical trunk and MF weights, evaluate the pressure field on CPU, and plot the casewise errors saved across the validation and held-out sets. |
| 9 | Plot the saved eight homogeneous inversions, two heterogeneous cases, and repeated noise trials. |
| 10 | Plot the three-seed component ablation and calibration-budget comparisons. |
| 11 | Read the 40 × 40 screening grid, the laboratory inverse-error curve and the recorded timing summary. The direct-solver sweep time in the summary is an estimate obtained by multiplying measured single-case time by 1,600. |

This command generates 25 publication panels and legends across nine figures. Use the separate flat-seabed benchmark workflow for Figures 3 and 4. Rendering curves from stored profiles tests a different stage of the workflow from recalculating the profiles by training.

## Comparison with the reference panels supplied with the manuscript

On Windows, all 25 regenerated PDFs were rasterized at 110 dpi with Poppler and matched the corresponding manuscript panel pixels using the rendering environments below. The comparison uses the same Times New Roman and STIX fonts as the manuscript. PDF creation metadata can differ between runs.

| Panels | Python | Matplotlib | NumPy | pandas |
|---|---|---|---|---|
| 1, 5, 6 | 3.12.4 | 3.10.6 | 2.3.5 | 2.3.2 |
| 2, 7, 8, 9, 10, 11 | 3.13 | 3.10.0 | 2.1.3 | 2.2.3 |

The Figure 8 replay used PyTorch 2.12.0 on CPU. On other systems the default plot style falls back to DejaVu Serif if Times New Roman is unavailable.

## Data provenance for the pipe-wall panels

The numeric reference columns in `legacy/figure9_mpm_kan/data/figure9.csv` and `legacy/figure15_mpm_kan/data/figure15{a,b}.csv` are digitizations of comparison curves from Figures 9 and 15 of [Zhu et al. (2024)](https://doi.org/10.1016/j.oceaneng.2024.119733). These figures include the published Turcotte, Liu and Kulhawy (1984) and [Sun et al. (2019)](https://doi.org/10.1016/j.oceaneng.2019.01.022) flume observations, alongside analytical or numerical curves from [Cheng and Liu (1986)](https://doi.org/10.1016/S0141-1187(86)80027-X), [Jeng (2001)](https://doi.org/10.1016/S0267-7261(01)00043-4) and Zhu et al. Experimental values are digitized coordinates rather than original instrument series. The SeaKAN prediction columns and model-comparison outputs are from the authors' calculations.
