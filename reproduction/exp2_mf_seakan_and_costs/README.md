# Parametric SeaKAN and MF-SeaKAN reproduction

Run the commands from the repository root with the reproduction dependencies installed. The published numerical assets in this directory are read-only inputs to the verification commands.

## Frozen weights, reference fields and inverse results

```bash
python reproduction/exp2_mf_seakan_and_costs/verify_tier2.py --chart --inversions --noise --report runs/tier2/verification.json
python reproduction/exp2_mf_seakan_and_costs/refine_turcotte_lab_inversion.py --use-published-profiles --output-dir runs/tier2/flume
```

The verifier defaults to CPU. With no switches it evaluates the published three trunk checkpoints and the selected calibrated checkpoint against the exported COMSOL fields. The optional switches recompute the entire 40 × 40 design chart using spatial automatic derivatives, ten numerical permeability inversions with eight starts per case, and the sensor-noise repetitions. The report contains observed differences and the tolerances used. Runtime depends on the device and thread settings.

Table 7 reports the selected trunk and the calibrated seed 202438. Table 11 reports a three-seed mean at each calibration budget. The selected eight-case amplitude error is 3.83%; the three-seed 32-field mean is 3.87%. Single-checkpoint evaluation reproduces the former. Fresh sample-efficiency training is required to independently regenerate every row of the latter.

The archived design grid uses a mudline pressure amplitude of 458.36 Pa. At hydraulic conductivity 1e-5 m/s, the peak gradient magnitude is 0.663, 0.812, and 0.881 at burial ratios 0.75, 1.25, and 1.50, respectively. The code evaluates the spatial gradient magnitude. These values matter when comparing the grid with the engineering interpretation in the manuscript.

For the Table 9 numbering, B1 and B2 are `blind_000` and `blind_001`, and B3 through B8 are internally named B6 through B11 (files `sealed_002` through `sealed_007`). B7-het and B8-het are internally B10_het and B11_het. These labels refer to the same parameter sets.

The flume command fits the eight transducer measurements from Turcotte et al. (1984) to cubic splines through nine archived, separately trained physics-only SeaKAN profiles. Omitting `--use-published-profiles` trains the nine profiles in the specified output directory; choose a new directory for a fresh run. The laboratory inversion uses this geometry-specific family. Numerical inversions use the frozen 32-field MF-SeaKAN.

## Training

```bash
python reproduction/exp2_mf_seakan_and_costs/retrain_mf32.py --output-dir runs/fresh-mf32
python reproduction/exp2_mf_seakan_and_costs/exp2_mf_seakan_and_costs.py
```

The first command trains the selected Table 7 configuration from scratch: 2,000 physics-only updates (seed 202418) followed by 800 calibration updates on 32 fields (seed 202438). It records the new weights and metrics in an unused directory. It defaults to CUDA; `--device cpu` is supported. GPU arithmetic and some parallel reductions can vary across runs.

The second command runs the full published experiment: three physics trunks, the pure-data and multi-fidelity models at 4/8/16/32 fields with three seeds each, inversions, external flume models, and GPU/CPU cost and mesh comparisons. The transferred trunk is selected using the COMSOL validation set's complex-field relative error. The complete benchmark requires a CUDA-enabled PyTorch build. By default it writes new results under `reproduction/exp2_mf_seakan_and_costs/runs/`. The environment variable `SEAKAN_OUTPUT_DIR` selects another output directory. Existing trunk and flume checkpoints there are reused; select a new directory to start fresh. Fresh run times are machine-specific.

## COMSOL references

The adjacent `comsol_clean_suite` directory includes the 52 high-fidelity numerical reference fields, 21 mesh-comparison fields, calibration-case parameters, the batch timing table, and the COMSOL model generator. Physical inputs and data provenance are described in the manuscript. CPU validation of the supplied fields requires only Python dependencies. To regenerate fields with licensed COMSOL Multiphysics 6.4, use a fresh directory, for example:

```bash
python reproduction/comsol_clean_suite/regenerate_comsol.py --work-dir runs/new-comsol
python reproduction/comsol_clean_suite/compare_fields.py runs/new-comsol/outputs
```

`--bin-dir` supplies the directory containing `comsolcompile` and `comsolbatch` if they are not on PATH (for example the Windows installation's `Multiphysics/bin/win64`). The Java export can also be compiled and launched directly with the COMSOL tools from a working directory with an `outputs` subdirectory. The wrapper explicitly supplies the output path, aggregates timing rows, and preserves solver logs. To use regenerated reference fields with the training or verification commands, set `SEAKAN_COMSOL_RUN` to the new run directory. The generation script performs 74 solves including one warm-up. The 1,600-case direct-FEM costs in the paper are extrapolated from the recorded per-solve mean; the network grid is timed as one batch sweep.
