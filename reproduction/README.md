# Reproduce the SeaKAN manuscript

Run these commands from the top of a checkout. The original SeaKAN library lives in `src/`. The versioned experiments here include their own model definitions so the supplied weights can be read with the architectures used to train them.

```sh
python -m pip install -e ".[reproduction,dev]"
python -m pytest
python reproduction/verify_benchmarks.py --output runs/benchmarks
python reproduction/verify_tier1.py --output runs/pipe
python reproduction/replay_tier1_models.py
python reproduction/exp2_mf_seakan_and_costs/verify_tier2.py --report runs/tier2/check.json
python reproduction/exp2_mf_seakan_and_costs/refine_turcotte_lab_inversion.py --use-published-profiles --output-dir runs/flume
```

These checks run on CPU with the supplied arrays and model weights. The baseline weight replay uses DeepXDE with its PyTorch backend. A tested environment used Python 3.12.4, PyTorch 2.13.0, NumPy 2.3.5, pandas 2.3.2, SciPy 1.16.3 and DeepXDE 1.14.0. A compatible CUDA installation accelerates training. Generating new finite-element fields requires a licensed COMSOL 6.4 installation. Reading the exported fields requires only Python.

| Research result | What the command evaluates |
|---|---|
| Flat and trenched seabed benchmarks (Figs. 3, 4 and 6) | `verify_benchmarks.py` calculates new predictions from the released weights for four source-study benchmarks. It also rescores 71 profile comparisons and checks 110 archived summary values. |
| Single buried pipe and component ablation (Tables 6 and 10) | `verify_tier1.py` evaluates the selected SeaKAN at 721 angles, scores all six models against digitized experimental and FEM references, and aggregates the 18 saved ablation profiles. `replay_tier1_models.py` evaluates the five available model weight sets. The original Standard KINN profile and a from-scratch trainer are supplied. [Training and baseline details](TIER1.md) |
| Parametric forward model and calibration (Tables 7 and 11) | `verify_tier2.py` evaluates the three archived physics trunks and the selected calibrated model. For the selected checkpoint the mean held-out amplitude error is 3.83%. The 32-field three-seed mean in Table 11 is 3.87%. [Training and seed selection](exp2_mf_seakan_and_costs/README.md) |
| Inversions, design chart and sensor noise | Add `--chart --inversions --noise` to `verify_tier2.py` to recompute the 1,600-point grid, ten numerical inversions and noise trials. The separate flume command fits eight measured amplitudes to a family of nine physics-only model profiles. |
| Independent FEM fields | `comsol_clean_suite/regenerate_comsol.py` generates 52 main fields and 21 mesh-check fields following one warm-up. `compare_fields.py` compares complex pressure at matching coordinates. [COMSOL commands](exp2_mf_seakan_and_costs/README.md#comsol-references) |
| Figures | `python reproduction/generate_revision_figures.py --output-dir runs/figures` generates 25 panels and legends for Figs. 1, 2 and 5–11. `python reproduction/plot_flat_benchmarks.py --output runs/flat` redraws eight panels for Figs. 3–4 using the benchmark arrays. The latter plots have a reconstructed layout. [Panel inputs and rendering environments](FIGURES.md) |

## Interpreting a replay

The experimental relative L2 error of 5.41% belongs to the selected buried-pipe model. The developers used pipe-crown observations, overall profile error and physics residuals to select hyperparameters and a checkpoint. The flume data were excluded from gradient updates, but the reported fit uses the same observations seen during model selection. The 10.91% reported for the full variant in the ablation comes from a separate three-seed series. The laboratory conductivity inversion uses a separate profile family and gives approximately 8.71 × 10⁻⁴ m/s, with a relative error of 20.83% against the reported conductivity.

An independent run on an NVIDIA RTX 4090 D started from random weights, trained the physical trunk for 2,000 updates and trained the 32-field corrector for 800 updates. On the eight held-out cases the mean amplitude error decreased from 10.0254% to 3.7181% (mean R² 0.9667 to 0.9955). Replaying the corresponding archived weights gave 10.3983% and 3.8346%. Separately, a full rerun of the 74 COMSOL 6.4 solves reproduced all 73 exported pressure fields with a maximum coordinate-matched relative complex-field L2 difference of 1.57 × 10⁻¹⁶ or less.

Two parameter counts in the reviewed Table 6 differ from the released weights. The XPINN contains 15,858 trainable scalar entries and the PINO contains 793,446 trainable tensor entries, counting a complex entry once. Table 6 lists 13,458 and 58,438, respectively. The released CPU PINO weights give experimental/FEM errors of 36.3685%/29.1802%, compared with 36.3665%/29.1715% for the archived prediction. The [buried-pipe guide](TIER1.md) describes the available training records.

The full training entry points are described in the [buried-pipe](TIER1.md) and [parametric](exp2_mf_seakan_and_costs/README.md) guides. Start new training in unused output directories. Training time depends on hardware, and a newly optimized model can differ from the saved checkpoint even with the same seed. Timing estimates for the 1,600-case direct-FEM sweep use extrapolated per-solve costs. Digitized measurements are comparison coordinates from the cited publications. References for the pipe-wall data are listed in the [figure guide](FIGURES.md#data-provenance-for-the-pipe-wall-panels).
