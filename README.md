# SeaKAN & MF-SeaKAN

Official code, COMSOL Multiphysics 6.4 benchmark suite, pre-trained checkpoints, and reproducibility scripts for:

> **Physics-Informed Kolmogorov--Arnold Networks for Wave--Seabed--Pipeline Interaction**

SeaKAN is a reference-free physics-informed solver for frequency-domain wave--seabed--pipeline Biot poroelasticity, paired with **MF-SeaKAN**, a multi-fidelity parametric surrogate calibrated on sparse COMSOL 6.4 fields for rapid stability chart screening and seabed permeability inversion.

---

The [manuscript reproduction guide](reproduction/README.md) provides the earlier training and evaluation workflows. The [framework usage guide](https://github.com/AiRzT/SeaKAN/blob/8c1774438c10699e24e6e775d8239bfde3fd0164/README.md) documents the core command-line tools.

## Repository Structure

```text
.
├── LICENSE                          # MIT License
├── pyproject.toml                   # Package metadata & CLI entry points
├── requirements.txt                 # Core Python dependencies
├── src/
│   ├── mpm_kan_generalization/      # Core parametric SeaKAN & inverse modules
│   ├── mpm_kan_figure9/             # Single-scenario SeaKAN + Hermite crown branch
│   ├── mpm_kan_figure15/            # Trenched-pipeline benchmark loader
│   └── figure9_comparison/          # Pointwise baseline comparison helpers
├── tests/                           # Unit tests for core physics, KAN, and MPM modules
├── data/
│   ├── benchmarks/                  # Digitized literature benchmarks & baseline profiles
│   └── comsol_clean_suite/          # COMSOL 6.4 Java batch suite, timings, and 52+ exported CSV fields
└── experiments/
    ├── exp1_kinn_and_ablation/      # Standard KINN baseline & 3-seed SeaKAN component ablation
    ├── exp2_mf_seakan_and_costs/    # MF-SeaKAN trunk, sample efficiency, inversion, 1600-case charts
    │   ├── exp2_mf_seakan_and_costs.py
    │   ├── refine_turcotte_lab_inversion.py
    │   └── recompute_table8_observability.py
    └── generate_revision_figures.py # Regenerates all vector PDF figures from saved outputs
```

---

## Installation

Python **3.11+** is required. A CUDA-capable GPU (e.g., NVIDIA RTX 4090 D) is recommended for retraining from scratch; CPU execution is supported for evaluation, inversion queries, and figure generation.

```bash
git clone https://github.com/AiRzT/SeaKAN.git
cd SeaKAN
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

Run unit tests to verify the installation:

```bash
python -m pytest
```

---

## Reproducing Manuscript Results

All Python scripts use **relative paths** and require no manual path edits. Exported COMSOL 6.4 reference fields (`data/comsol_clean_suite/outputs/*.csv`) and pre-trained MF-SeaKAN checkpoints (`clean_trunk_seed_202418.pt`, `selected_mf32_model.pt`) are included so you can reproduce all evaluations without a COMSOL license.

### 1. Standard KINN & 3-Seed Component Ablation (Tables 6 & 10, Fig. 7 & 10a)

```bash
python experiments/exp1_kinn_and_ablation/exp1_kinn_and_ablation.py
```

- Runs the pointwise strong-form `KINN` baseline (`13,131` parameters, seed `202409`) and the 3-seed (`202409`–`202411`) `SeaKAN` component ablation on the Turcotte et al. (1984) buried-pipeline benchmark.
- Outputs `exp1_complete_results.json` and `ablation_summary.csv`.
- *Note*: Table 5/6's `5.41%` `SeaKAN` profile is archived in `data/benchmarks/figure9_top_gap_augmented_profiles.csv` (from the multi-stage continuation checkpoint), whereas `exp1_kinn_and_ablation.py` trains all six ablation variants from fresh random initialisation under a uniform single-script budget (`10.91 ± 2.73%`, best seed `7.75%`).

### 2. Physics Trunk, MF-SeaKAN Sample Efficiency, Inversion & 1,600-Case Charts (Tables 7, 9, 11, 12; Figs. 8, 9, 10b, 11)

```bash
python experiments/exp2_mf_seakan_and_costs/exp2_mf_seakan_and_costs.py
```

- Trains the zero-data 1st-order MPM weak-form parametric trunk (seeds `202416`–`202418`), evaluates `MF-SeaKAN` vs. pure-data KAN across $N_{\mathrm{cal}}\in\{4,8,16,32\}$ (seeds `202438`–`202440`), runs 64-sensor Fisher permeability inversion (B1–B8, `0%–5%` noise, $\mathrm{COV}_k=0.35$ heterogeneous tests), and evaluates the $40\times 40 = 1{,}600$ $(k,b/D)$ design chart.

### 3. Observability & Jacobian Condition-Number Screening (Table 8)

```bash
python experiments/exp2_mf_seakan_and_costs/recompute_table8_observability.py
```

- Evaluates `selected_mf32_model.pt` across all 8 held-out COMSOL 6.4 cases (B1–B8) comparing 64 pipe-wall amplitude gauges vs. 64 Fisher-selected interior complex-pressure gauges (joint $(k,b/D)$ vs. known $b/D$ inverting $k$ only).

### 4. External Wave-Flume Permeability Inversion (Table 9 & Fig. 11c)

```bash
python experiments/exp2_mf_seakan_and_costs/refine_turcotte_lab_inversion.py
```

- Solves 9 Chebyshev-node physics-only `SeaKAN` models over $k\in[3\times 10^{-4},\,3\times 10^{-3}]~\mathrm{m\,s^{-1}}$ for the Turcotte et al. (1984) flume geometry (independent of COMSOL) and inverts $k$ from the 8 wall-transducer measurements.

### 5. Regenerate All Manuscript Vector Figures

```bash
python experiments/generate_revision_figures.py
```

- Writes all vector PDFs to `experiments/reproduced_figures/`.

### 6. (Optional) Re-run COMSOL Multiphysics 6.4 Reference Suite

Requires a licensed installation of **COMSOL Multiphysics 6.4**:

```powershell
cd data\comsol_clean_suite
& "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolcompile.exe" CleanSealedBiotSuite.java
& "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64\comsolbatch.exe" -inputfile CleanSealedBiotSuite.class
```

---

## Reproducibility Notes

- **Random seeds**: Fixed in all scripts (`202409–202411` for Exp. 1; `202416–202418` for the physics trunk; `202438–202440` for `MF-SeaKAN` calibration).
- **Hardware & determinism**: Reference timings in Table 12 were measured on an AMD Ryzen 9 9950X3D CPU and NVIDIA RTX 4090 D GPU. Minor floating-point differences may occur across different CUDA/PyTorch versions or GPU architectures, though statistical trends and engineering conclusions remain unchanged.

---

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{Cai2026SeaKAN,
  author  = {Erxuan Cai and Jing Li and Zhiran Gao and Zhexi Xu and Lewei Feng and Minjie Wen and Yiming Zhang},
  title   = {Physics-Informed {Kolmogorov--Arnold} Networks for Wave--Seabed--Pipeline Interaction},
  journal = {Applied Ocean Research},
  year    = {2026},
  note    = {Under review; code and data at https://github.com/AiRzT/SeaKAN}
}
```
