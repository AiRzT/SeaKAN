# Training the flat-seabed and trench benchmarks

The folder numbers refer to figures in Zhu et al. (2024). Source figures 7, 8, 13 and 15 correspond to SeaKAN manuscript Figures 3, 4a, 4b–c and 6, respectively. Each benchmark has its own model and checkpoints. Start the PowerShell commands below at the repository root after installing the reproduction dependencies. Use a new output directory for each experiment.

To recompute the released curves and scores on CPU, run `python reproduction/verify_benchmarks.py --output runs/benchmarks`. To redraw Figures 3 and 4 from the archived profiles, run `python reproduction/plot_flat_benchmarks.py --output runs/flat_figures`.

## Figure 3 · Four flat-seabed cases

The following independent single-thread CPU run started from random weights and used the full coarse-sand budget. It completed in 354 s on Python 3.12.4 and PyTorch 2.13.0. Relative L2 pore-pressure error was 0.504866% against the analytical data and 0.166078% against the numerical data. The released weights gave 0.515450% and 0.177137%, respectively. Across four fields and two references, the largest absolute difference was 0.316790 percentage points, for shear stress against the numerical profile. These figures describe one run.

```powershell
$fig7 = 'reproduction/legacy/src/run_fig7.py'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python $fig7 --device cpu --cases saturated_coarse --seed 202407 --steps 1600 --lbfgs-steps 150 --log-every 400 --output-dir runs/flat_coarse
```

Replace `saturated_coarse` with `unsaturated_coarse` to train the other coarse-sand case. For fine sand, the archived runs included several optimizer restarts. The two supplied initial checkpoints contain the original 1,600-step weights before strong-form stabilization was added. The following commands reconstruct the continuation schedule in the saved histories. The arguments `--fine-steps` are cumulative Adam targets. Run each command in order so that `--resume` can find the previous file.

```powershell
$fine = 'runs/flat_fine'
$initial = 'reproduction/legacy/src/outputs/figure7/initial_checkpoints'
New-Item -ItemType Directory -Path "$fine/checkpoints" | Out-Null
Copy-Item -Path "$initial/*.pt" -Destination "$fine/checkpoints"
python $fig7 --device cuda --output-dir $fine --cases saturated_fine unsaturated_fine --resume --fine-steps 3200 --fine-lbfgs-steps 250 --log-every 400
python $fig7 --device cuda --output-dir $fine --cases saturated_fine --resume --fine-steps 5000 --fine-lbfgs-steps 400 --learning-rate 0.0001 --grid-cells 64 --momentum-weight 10 --strong-weight 0.5 --roughness-weight 0.000001
python $fig7 --device cuda --output-dir $fine --cases unsaturated_fine --resume --fine-steps 5000 --fine-lbfgs-steps 400 --learning-rate 0.0002 --grid-cells 64 --momentum-weight 1 --strong-weight 0.5 --roughness-weight 0.000001
python $fig7 --device cuda --output-dir $fine --cases unsaturated_fine --resume --fine-steps 6500 --fine-lbfgs-steps 400 --learning-rate 0.0001 --grid-cells 64 --momentum-weight 10 --strong-weight 0.5 --roughness-weight 0.000001
python $fig7 --device cpu --output-dir "$fine/profiles" --checkpoint-dir $fine --cases saturated_fine unsaturated_fine --evaluate-only
```

To initialize both fine-sand runs with new random weights in an unused `$fine` directory, replace the checkpoint copy with `python $fig7 --device cuda --cases saturated_fine unsaturated_fine --output-dir $fine --fine-steps 1600 --fine-lbfgs-steps 150 --strong-weight 0 --log-every 400`. The first historical run predates the strong-residual option. The original 1,600-step snapshots preserve that starting point for the continuation. Digitized curves informed fine-sand checkpoint selection after optimization. This release check exercised the coarse-sand cold start; the four-stage fine-sand schedule was recovered from the recorded configurations and optimizer histories.

## Figure 4 · Additional flat-seabed benchmarks

Source figure 8 trains in two stages in one invocation. The default first stage uses 3,200 Adam and 250 L-BFGS steps. The refinement uses 5,000 Adam as the cumulative target and 400 L-BFGS steps on 64 cells.

```powershell
$fig8 = 'reproduction/legacy/figure8_mpm_kan'
$env:PYTHONPATH = (Resolve-Path "$fig8/src").Path
python "$fig8/run_figure8.py" --device cuda --output-dir runs/flat8
```

For source figure 13, the historical run used three successive stages. The final stage raises the momentum residual weight to 100.

```powershell
$fig13 = 'reproduction/legacy/figure13_mpm_kan'
$env:PYTHONPATH = (Resolve-Path "$fig13/src").Path
python "$fig13/run_figure13.py" --device cuda --output-dir runs/flat13
python "$fig13/run_figure13.py" --device cuda --output-dir runs/flat13 --checkpoint runs/flat13/checkpoints/figure13.pt --resume --resume-adam-steps 240 --resume-lbfgs-steps 80
python "$fig13/run_figure13.py" --device cuda --output-dir runs/flat13 --checkpoint runs/flat13/checkpoints/figure13.pt --resume --resume-adam-steps 320 --resume-lbfgs-steps 100 --resume-momentum-weight 100
```

The release check reran both Figure 4 solvers from the supplied weights. The training commands above follow the archived run configurations.

## Figure 6 · Trenched pipe

The [trench training guide](figure15_mpm_kan/REPRODUCTION.md) gives the water, soil and contact training stages. Section 6 gives the final 30-step CUDA wall-model refinement and assembly. It uses a standalone RANS starting model and a separate joint checkpoint that supplies the soil and porous interface. For the release check, the already-trained historical wall model was assembled with that joint checkpoint on CPU. All 132 tensor entries matched the released joint model. Evaluation gave 8.292677% error on Test 49 experimental points and 3.544513% on the combined numerical interface. The 30-step CUDA optimizer settings were checked against the historical run record. This release check did not repeat those 30 updates.

Figure 5 and Tables 6 and 10 use a separate buried-pipe experiment, documented in the [buried-pipe training guide](../TIER1.md).
