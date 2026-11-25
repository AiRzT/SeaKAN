# SeaKAN

SeaKAN is a physics-only, parameter-conditioned framework for frequency-domain wave--seabed interaction. It combines a Kolmogorov--Arnold network (KAN) surrogate with material-point residual evaluation, particle-to-grid (P2G) aggregation, hard boundary transforms, augmented-Lagrangian boundary constraints, random-field conditioning, and frozen-model inverse analysis.

The [manuscript reproduction guide](reproduction/README.md) contains the experiment-specific training and evaluation scripts, numerical reference fields, digitized comparison points, released model weights and figure generators. The core library and its command-line interfaces are described below.

## Framework modules

- `kan.py` and `model.py`: KAN layers, conditional feature encoding, local branch, and complex-valued field outputs.
- `mpm.py`: particle clouds, local refinement, boundary particles, and P2G transfer.
- `physics.py`: strong-form poroelastic residuals and augmented-Lagrangian constraints.
- `random_field.py`, `cases.py`, and `design.py`: random-field conditioning and disjoint parameter-case design.
- `training.py` and `cli.py`: physics-only optimisation, robust case aggregation, checkpointing, and frozen evaluation.
- `inverse.py`, `field_inverse.py`, and `comsol_calibration.py`: reusable parameter-inversion and optional multi-fidelity calibration routines.
- `metrics.py` and `scoring.py`: relative L2 error, MSE, RMSE, and R2 scoring.

## Installation

SeaKAN requires Python 3.11 or newer. CUDA is used when a compatible PyTorch installation is available; CPU execution is also supported.

```bash
git clone https://github.com/AiRzT/SeaKAN.git
cd SeaKAN
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Basic use

Inspect the complete training interface:

```bash
seakan --help
```

Run a small physics-only smoke calculation:

```bash
seakan --device auto --steps 2 --float64-steps 0 --train-cases 4 --validation-cases 2 --blind-cases 2 --output-dir outputs/smoke
```

The standard CLI creates a disjoint train/validation/blind design, trains one conditional model, freezes it, and evaluates the blind cases without retraining. Output directories are ignored by Git and remain local.

Inverse-analysis commands require a user-supplied checkpoint and observation file:

```bash
seakan-inverse --help
seakan-field-inverse --help
```

## Reproducing the results

Install the experiment dependencies with `python -m pip install -e ".[reproduction]"`. Start with the CPU commands in the [reproduction guide](reproduction/README.md). The guide distinguishes evaluation of released weights, scoring of saved predictions, independent training, and regeneration of finite-element references. It also documents discrepancies found when checking the manuscript.

## Citation

If you use this framework, cite the accompanying SeaKAN manuscript and link to this repository:

<https://github.com/AiRzT/SeaKAN>
