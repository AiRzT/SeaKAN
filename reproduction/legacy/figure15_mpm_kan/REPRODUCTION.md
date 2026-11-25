# 重训沟槽管道模型与复核发布检查点

本目录对应修订稿图 6 和表 5，文件名中的 15 是 Zhu 等（2024）的原图号。重算已发布结果可直接跳到第 7 步。要复核最后一次管壁训练，从第 6 步的两份历史权重开始。需要研究早期训练时，再运行第 1–5 步。这些步骤沿用 v1.3 项目的历史构建流程，这次核验没有从随机初值跑完整个流程。

训练损失仅包含物理方程、边界与界面残差。`figure15a.csv`、`figure15b.csv` 和它们的评分用于训练后的评估。旧版候选筛选也比较过这些评分，包括不同耦合方式对埋入段孔压误差的影响。

在 `reproduction/legacy/figure15_mpm_kan` 下运行。训练步骤需要支持 CUDA 的 PyTorch，组装和评估支持 CPU。

```powershell
$py = "python"
$env:PYTHONPATH = (Resolve-Path "src").Path
```

## 1. 基础势流–土体检查点

发布包已附带 `artifacts/checkpoints/figure15_base_v1_1.pt`。需要完全重建时：

```powershell
& $py run_figure15.py --device cuda `
  --output-dir outputs/base_physics `
  --water-steps 600 --water-float64-steps 60 --water-lbfgs-steps 30 `
  --teacher-steps 800 --teacher-anchor-steps 100 `
  --soil-steps 1200 --soil-float64-steps 120 --soil-lbfgs-steps 50
```

## 2. 训练 Test 49 固壁谐波 RANS MPM-KAN

```powershell
& $py scripts/train_rans_water.py `
  --output outputs/rans_fp32 --steps 160 --float64-steps 0 `
  --learning-rate 0.0005 --effective-viscosity 0.00015 `
  --width 16 --hidden-layers 2 --basis 9 --seed 202449 --log-every 20
& $py scripts/train_rans_water.py `
  --initial-checkpoint outputs/rans_fp32/water_test49_rans.pt `
  --output outputs/rans_wall --steps 0 --float64-steps 40 `
  --learning-rate 0.0003 --effective-viscosity 0.00015 `
  --width 16 --hidden-layers 2 --basis 9 --seed 202449 --log-every 10
```

该阶段训练强形式谐波不可压缩动量与连续方程、自由面条件、硬无滑移和远场入射波，
并使用第四级近管边界层粒子加密及 P2G 残差。

## 3. 组装初始分区检查点

```powershell
& $py scripts/assemble_partitioned_checkpoint.py `
  --base-checkpoint artifacts/checkpoints/figure15_base_v1_1.pt `
  --rans-checkpoint outputs/rans_wall/water_test49_rans.pt `
  --output outputs/partitioned/checkpoints/figure15.pt
```

此检查点的暴露管壁压力来自 RANS-KAN，多孔海床基础载荷来自势流 KAN。
组装脚本不读取图 15 CSV。

## 4. 训练局部接触 KAN

```powershell
& $py scripts/train_contact_enrichment.py `
  --checkpoint outputs/partitioned/checkpoints/figure15.pt `
  --output outputs/contact_enriched `
  --steps 100 --float64-steps 20 `
  --learning-rate 0.0005 `
  --contact-weight-start 2000 `
  --contact-weight-end 50000 `
  --angle-scale 3.0
```

两个局部 KAN 仅通过水域物理残差与 RANS/多孔界面的复压力连续残差训练。
`angle-scale=3.0` 对应 `30.047°` 紧支撑角。其五次权函数只是网络结构中的局部支撑，
不在预测或绘图阶段修改数据。

## 5. 在增强界面下续训强 Biot Soil-KAN

```powershell
& $py scripts/continue_enriched_soil.py `
  --checkpoint outputs/contact_enriched/checkpoints/figure15.pt `
  --output outputs/release_candidate `
  --steps 160 --float64-steps 30 `
  --learning-rate 0.0001 `
  --float64-learning-rate 0.00001 `
  --train-global
```

接触增强由界面向土内按其垂向支撑尺度指数衰减，然后训练全局和近管 Soil-KAN。
这一步避免把局部接触增压沿整个土层硬复制。

## 6. 从 v1.3 权重续训 30 步并组装最终检查点

`artifacts/checkpoints/water_test49_before_wall_refinement.pt` 保存了旧版独立 RANS 水模型。其来源是步骤 2 对应的历史运行 `candidate32_rans_float64`。归档原件与旧 v1.3 独立水权重的 SHA-256 相同。`artifacts/checkpoints/figure15_before_wall_refinement.pt` 是 v1.3 选定的 `candidate53` 联合模型，含两种工况的土体网络和带接触增强的多孔界面。独立水权重用于优化暴露管壁，联合权重用于提供最后组装时继承的场。发布副本清除了原工作机的绝对路径，保留了每个张量及其精度。

历史运行 `candidate56a_wall_physics` 将管壁动量权重从 0.5 提高到 10。优化器执行 30 步单精度 Adam，随后在双精度下评估全体材料点。运行以下命令可从两份历史输入重复最后阶段。

```powershell
& $py scripts/train_rans_water.py `
  --initial-checkpoint artifacts/checkpoints/water_test49_before_wall_refinement.pt `
  --output outputs/refined_wall --steps 30 --float64-steps 0 `
  --learning-rate 0.0001 --effective-viscosity 0.00015 `
  --width 16 --hidden-layers 2 --basis 9 `
  --continuity-weight 8 --free-surface-weight 80 `
  --wall-momentum-weight 10 --far-field-weight 1200 --seed 202449 --log-every 10
& $py scripts/assemble_partitioned_checkpoint.py `
  --base-checkpoint artifacts/checkpoints/figure15_before_wall_refinement.pt `
  --rans-checkpoint outputs/refined_wall/water_test49_rans.pt `
  --output outputs/final/checkpoints/figure15.pt
```

如要串联第 1–6 步的本次新训练，可在上面两条命令中分别用 `outputs/rans_wall/water_test49_rans.pt` 和 `outputs/release_candidate/checkpoints/figure15.pt` 替换两份发布的历史输入；第 3 步的 `--base-checkpoint` 同时指向第 1 步产生的 `outputs/base_physics/checkpoints/figure15.pt`。新优化的结果会随训练环境变化。

## 7. 独立验证和绘图

```powershell
& $py run_figure15.py --device cpu --evaluate-only `
  --checkpoint artifacts/checkpoints/figure15.pt `
  --output-dir outputs/release_validation
```

要复核第 6 步新组装的文件，将 `--checkpoint` 改为 `outputs/final/checkpoints/figure15.pt`。主要输出如下。

- `figure15_mpm_kan_pinn.png`
- `figure15_metrics.json`
- `figure15_profiles.csv`
- `run_config.json`
- `training_history.json`

`figure15_metrics.json` 的 `contact_continuity` 同时记录原始复压力跳变和
`postprocessing_applied: false`；`run_config.json` 记录接触 KAN 架构与训练来源。

本次核验实际在 CPU 上使用历史 `candidate56a` 训练后的水模型和公开的 v1.3 联合输入重新组装。所得模型与最终发布模型的 132 项张量逐项相同。独立评估给出 Test 49 的实验点误差 8.292677%、整圈数值界面误差 3.544513%、埋入弧和暴露弧数值误差分别为 5.858715% 和 1.903527%。30 步 CUDA 优化的参数已对照历史运行记录，此次未重新执行该优化。
