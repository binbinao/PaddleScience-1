# Sedan Aero AI

本案例使用 3D 傅里叶神经算子（FNO）构建轿车外气动 AI 替代模型，以车辆几何的符号距离函数（SDF）为输入，直接预测周围流场 $(u, v, w, p)$，相比传统 RANS CFD 仿真实现多个数量级的加速。

本案例为独立实现（argparse + YAML 配置），不依赖 PaddleScience 的 Hydra 配置体系；安装 PaddlePaddle 后即可运行。

=== "模型训练命令"

    ``` sh
    cd examples/operator_learning/sedan_aero_ai
    # FNO 架构（默认），合成数据自动生成
    python train.py --model fno
    # 3D CNN 架构
    python train.py --model cnn
    # 覆盖训练超参
    python train.py --model fno --epochs 500 --batch-size 4 --lr 5e-4
    ```

=== "模型评估命令"

    ``` sh
    # 复用训练输出目录中已生成的验证数据
    python train.py --model fno --eval --output-dir ./outputs
    # 或指定权重路径
    python train.py --model fno --eval --pretrained ./outputs/final_model.pdparams --output-dir ./outputs
    ```

=== "模型推理命令"

    ``` sh
    # 加载训练好的模型推理 + 速度 benchmark
    python infer.py --model fno --model-path ./outputs/final_model.pdparams --benchmark
    # 多工况（不同来流速度）预测
    python infer.py --model fno --model-path ./outputs/final_model.pdparams --u-inf 25 30 35
    ```

注：训练完成后会自动运行一次推理 benchmark，并将结果保存至 `outputs/benchmark.json` 与 `outputs/prediction.npz`。

## 1. 背景简介

轿车外气动开发依赖大量 CFD 仿真：单次 RANS 计算在 32 核上约需 30 分钟，LES 更是长达数十小时，难以支撑造型迭代所需的大规模工况扫描。神经网络算子学习（operator learning）为该问题提供了替代路径：以几何表征为条件，直接学习"几何 → 流场"的映射，将推理成本压缩到毫秒级。

[FNO](https://export.arxiv.org/pdf/2010.08895.pdf) 通过在傅里叶域对低频模态做可学习的复数线性变换，兼具全局感受野与网格分辨率不变性，适合作为流场替代模型的主干。

当前实现使用**合成流场数据**（势流 + 尾流的经验模型）打通训练管线；接入真实 CFD 数据的方法见 [3.8 节](#38-接入真实-cfd-数据)。

## 2. 问题定义

给定参数化轿车几何 $\Omega_{car}$ 与来流条件（来流速度 $U_\infty$、雷诺数 $Re$），求解其周围三维流场：

$$
\begin{equation}
\mathcal{G}: \big(\text{SDF}(\mathbf{x}),\ \mathbf{x}\big) \mapsto \big(u,\ v,\ w,\ p\big)(\mathbf{x}), \quad \mathbf{x} \in \Omega \setminus \Omega_{car}
\end{equation}
$$

其中几何以符号距离函数 SDF 表示（车内为负、车外为正），计算域为长方体包围盒（默认 40m × 16m × 8m，车辆居中）。训练目标是最小化预测流场与参考流场（CFD 或合成数据）之间的均方误差。

默认物理参数：$Re = 2\times10^6$，$U_\infty = 30\,\mathrm{m/s}$（约 108 km/h 巡航），车长 $L_{ref} = 4.5\,\mathrm{m}$，迎风面积 $A_{ref} = 2.2\,\mathrm{m^2}$。

## 3. 问题求解

本案例为独立实现，关键组件组织在 `examples/operator_learning/sedan_aero_ai/` 下，通过 `sys.path.insert` 导入同级 `models/`、`data/`、`utils/` 包。

### 3.1 数据集构建

`data/generate.py` 提供两类能力：

1. **参数化轿车几何**（`SedanGeometry`）：以分段轮廓（引擎盖 → 风挡 → 车顶 → 后窗 → 行李厢）+ 车轮椭球构造 SDF，`get_sdf` 对包围盒网格求值。

2. **合成流场生成**（`FlowFieldGenerator`）：基于简化势流 + 尾流亏损模型生成带物理结构的流场（车面减速、尾流速度亏损、上洗/下洗、尾流区压力亏损）。注意：合成数据仅用于打通管线，不具备物理精度。

``` py linenums="296"
--8<--
examples/operator_learning/sedan_aero_ai/data/generate.py:296:303
--8<--
```

训练样本按 `DATA.train_samples` / `val_samples` 数量生成，来流速度在给定区间内随机采样；随后 `create_fno_dataset` 把样本堆叠为 FNO 张量格式：

- 输入：`(N, 1, D, H, W)` —— 单通道 SDF 体素
- 输出：`(N, 4, D, H, W)` —— $(u, v, w, p)$ 四通道流场

### 3.2 模型构建

主模型 `SedanAeroFNO`（`models/cfd_surrogate.py`）为自包含的 3D FNO 实现（安装 PaddleScience 后也可直接使用 `ppsci.arch.TFNO3dNet`）。每层 FNO 由三部分构成：

$$
x_{l+1} = \sigma\big(\mathcal{F}^{-1}(R_l \cdot \mathcal{F}(x_l)) + W_l x_l\big)
$$

即傅里叶域低频模态的可学习复数变换 + 1×1×1 卷积捷径：

``` py linenums="169"
--8<--
examples/operator_learning/sedan_aero_ai/models/cfd_surrogate.py:169:205
--8<--
```

备选模型 `SedanAeroCNN` 为带跳连的 3D U-Net，适合快速原型验证。通过 `--model {fno,cnn}` 选择，模型结构参数（模态数、隐藏通道数、层数等）在 `configs/sedan_fno.yaml` 的 `MODEL` 段配置。

### 3.3 超参数与优化器

训练默认 500 epochs、batch size 4，损失函数为 MSE；优化器 Adam（初始学习率 1e-3，weight decay 1e-4）配合 `StepDecay` 阶梯衰减（每 100 epochs 衰减 0.5）：

``` py linenums="227"
--8<--
examples/operator_learning/sedan_aero_ai/train.py:227:237
--8<--
```

所有超参均可通过 CLI 覆盖（`--epochs/--batch-size/--lr`），详见 `python train.py -h`。

### 3.4 训练流程

`train.py` 按以下顺序执行：生成（或加载）数据集 → 构建模型与优化器 → 逐 epoch 训练，每 `TRAIN.eval_freq` 个 epoch 在验证集上评估并记录最优值，每 `TRAIN.save_freq` 个 epoch 保存 checkpoint（`CheckpointManager` 最多保留 5 份）。训练结束后保存 `final_model.pdparams`、`metrics.json`，并自动执行一次推理 benchmark。

### 3.5 推理与气动系数估计

`infer.py` 加载训练权重后支持：

- **单/多工况预测**（`--u-inf 25 30 35`）：对每个来流速度生成测试几何并预测流场；
- **速度 benchmark**（`--benchmark`）：预热 3 次后计时 10 次取均值，输出吞吐量及相对 RANS（30 分钟）/LES（48 小时）的加速比；
- **气动系数估计**：基于预测压力场在车身表面（SDF 接近零处）数值积分估算 $C_d$、$C_l$。

结果输出到 `--output-dir`（默认 `./inference_results`）：`prediction.npz`（流场 + 坐标 + SDF）、`benchmark.json`、`aerodynamics.json`。

### 3.6 结果展示

在合成数据粗网格（16×8×4）CPU 环境下实测：单次 FNO 推理约 9 ms（约 108 samples/s），CNN 约 1.6 ms；相对 RANS CFD（30 分钟）加速比达 $10^5$ 量级。该数值仅反映管线速度，物理精度需接入真实 CFD 数据后重新评估。

### 3.7 目录结构与配置

```
sedan_aero_ai/
├── configs/sedan_fno.yaml    # 模型/物理/训练全参数配置
├── data/generate.py          # 轿车几何 SDF + 合成流场生成
├── models/cfd_surrogate.py   # SedanAeroFNO + SpectralConv3d + SedanAeroCNN
├── utils/helpers.py          # 指标/checkpoint/速度对比工具
├── train.py                  # 训练 / 评估入口
├── infer.py                  # 推理 + benchmark 入口
└── requirements.txt          # paddlepaddle, numpy, pyyaml
```

核心配置段：`MODEL`（FNO 结构）、`PHYSICS`（$Re$、$U_\infty$、$\rho$、$\nu$、参考面积）、`DOMAIN`（包围盒与网格分辨率）、`TRAIN`（epochs/batch/lr 调度）、`DATA`（合成数据开关、样本数、真实数据目录）。

### 3.8 接入真实 CFD 数据

1. 用 OpenFOAM / SU2 等完成 RANS 计算，导出每个几何的流场数据；
2. 将 `configs/sedan_fno.yaml` 中 `DATA.use_synthetic` 置为 `false`；
3. 准备 `train.npz` / `val.npz`（字段：`input` 形状 `(N, 1, D, H, W)` 的 SDF，`output` 形状 `(N, 4, D, H, W)` 的 $(u,v,w,p)$），放入 `DATA.data_dir` 目录。

## 4. 完整代码

``` py linenums="1" title="train.py"
--8<--
examples/operator_learning/sedan_aero_ai/train.py
--8<--
```

``` py linenums="1" title="infer.py"
--8<--
examples/operator_learning/sedan_aero_ai/infer.py
--8<--
```

``` py linenums="1" title="models/cfd_surrogate.py"
--8<--
examples/operator_learning/sedan_aero_ai/models/cfd_surrogate.py
--8<--
```

``` py linenums="1" title="data/generate.py"
--8<--
examples/operator_learning/sedan_aero_ai/data/generate.py
--8<--
```

## 5. 参考文献

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://export.arxiv.org/pdf/2010.08895.pdf)
- [Neural Operator: Learning Maps Between Function Spaces](https://export.arxiv.org/pdf/2108.08481.pdf)
- [TFNO3dNet — PaddleScience API](https://paddlescience-docs.readthedocs.io/zh-cn/latest/api/arch/)
