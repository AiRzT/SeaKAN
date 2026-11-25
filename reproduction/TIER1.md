# Buried-pipe benchmark reproduction

Run from the repository root. These commands evaluate Table 6 and the three-seed component ablation in Table 10. Their respective full-model experimental errors of 5.41% and 10.91% refer to a selected checkpoint and a separate three-seed training series.

```sh
python reproduction/verify_tier1.py --output results/table-replay
python reproduction/replay_tier1_models.py
python reproduction/tier1_smoke.py --output results/kinn-smoke
```

The first command computes predictions from the single-scenario SeaKAN weights at 721 pipe-wall angles, interpolates at the 8 laboratory and 163 finite-element reference angles, and checks relative L2 and R² at manuscript precision. It scores each of the other five archived predicted profiles in the same way. Table 10 is calculated from the 18 individual trained profiles (six variants × seeds 202409, 202410 and 202411), with sample standard deviations. The full-size tabulated results are independent of the three-update smoke test.

The second command runs five released models on CPU, using arrays of network tensors exported from the archived checkpoints. It requires DeepXDE 1.14.0 with its PyTorch backend. For the archived FP32/complex64 PINO, CPU and original GPU inference differ slightly: the maximum difference in normalized circumferential pressure measured here was 0.000160. CPU experimental and FEM errors were 36.3685% and 29.1802%, versus 36.3665% and 29.1715% for the archived profile. Standard KINN is supplied as a profile and full training script; its original trained weights were unavailable.

## Training

```sh
python reproduction/train_standard_kinn.py --device cuda --output results/kinn-full
python reproduction/exp1_kinn_and_ablation/exp1_kinn_and_ablation.py
python reproduction/train_tier1_baselines.py --output results/legacy-baselines
```

Standard KINN uses 1,200 Adam updates and 400 FP32/120 FP64 L-BFGS iterations; use a new output directory. The experiment-1 entry point runs Standard KINN and all six three-seed ablations, defaulting to one worker. It resumes completed runs and shared warm-start stages from `reproduction/exp1_kinn_and_ablation/runs`. Set `SEAKAN_EXP1_OUTPUT` to an empty directory for a fresh run. The single-scenario selected SeaKAN model in Table 6 is distinct from the Full SeaKAN variant retrained in Table 10.

The legacy baseline entry point runs 1,200 Adam updates for pointwise networks and 500 epochs for PINO using 26 finite-difference training fields; it requires a CUDA-enabled PyTorch environment. Its separate `--refine-only --methods deepxde_pinn gpinn_rar xpinn --lbfgs-steps 400` mode refines existing pointwise checkpoints in FP64. The archive records a 400-step FP64 tail for these models; the precise FP32/FP64 sequence implied by the manuscript cannot be confirmed for all legacy baselines. Existing scalar metrics are reproduced through the profile and weight replay above.

## Data and program versions

- Numeric comparison targets were digitized from Zhu et al. (2024), *Oscillatory responses of a spatially random seabed in the vicinity of structures: Validation and application of the integrated CFD–SFEM*, Figure 9, including the experimental data of Turcotte et al. (1984). [Reference article](https://doi.org/10.1016/j.oceaneng.2024.119733)
- `tier1_benchmarks/profiles` and `tier1_benchmarks/ablation` contain the authors' predicted angular profiles. `*.npz` files contain arrays of named PyTorch model tensors without optimizer state or local filesystem metadata.
- Parameter counts from the archived models are **15,858 for XPINN** and **793,446 for PINO** (PyTorch tensor-element convention, counting each complex-valued weight as one element). They differ from 13,458 and 58,438 in the reviewed Table 6. The replay program reports the counts derived from the weights.
- Local profile/weight replay used Python 3.12.4, PyTorch 2.13.0, DeepXDE 1.14.0, NumPy 2.3.5, SciPy 1.16.3 and pandas 2.3.2. The weight-replay scripts use CPU. The Standard KINN smoke test independently exercises the loss, gradient updates and prediction from a fresh initialization.
