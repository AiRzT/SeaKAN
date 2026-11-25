# 从头复现 v1.3 检查点

以下流程不会把 `figure15a.csv` 或 `figure15b.csv` 传入训练损失。
CSV 只在最后的 `--evaluate-only` 阶段读取。

在项目根目录设置环境：

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
  --output outputs/rans_wall `
  --steps 160 --float64-steps 40 `
  --effective-viscosity 0.00015 `
  --width 16 --hidden-layers 2
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

## 6. 独立验证和绘图

```powershell
& $py run_figure15.py --device cuda --evaluate-only `
  --checkpoint outputs/release_candidate/checkpoints/figure15.pt `
  --output-dir outputs/release_validation
```

此时才读取验证 CSV，并输出：

- `figure15_mpm_kan_pinn.png`
- `figure15_metrics.json`
- `figure15_profiles.csv`
- `run_config.json`
- `training_history.json`

`figure15_metrics.json` 的 `contact_continuity` 同时记录原始复压力跳变和
`postprocessing_applied: false`；`run_config.json` 记录接触 KAN 架构与训练来源。

## 7. 接受标准

发布候选必须同时满足：

1. 检查点元数据确认图 15 CSV 未进入训练；
2. 独立进程能重载检查点；
3. 两接触点原始复压力跳变均小于 `2e-4`；
4. 极坐标曲线无断裂、角度翻转或上下半圆拼接错误；
5. Test 49 论文孔压曲线相对 L2 小于 6%；
6. 物理残差有限，硬无滑移达到数值精度；
7. 单元测试和安装后评估均通过。
