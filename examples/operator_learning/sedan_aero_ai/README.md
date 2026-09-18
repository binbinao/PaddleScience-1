# Sedan External Aerodynamics AI Surrogate Model

基于 PaddlePaddle 的轿车外气动 CFD AI 替代算子，使用 TFNO3dNet (3D Fourier Neural Operator) 实现相比传统 CFD 百倍至千倍的流场预测加速。

## Architecture

| 组件 | 选择 | 说明 |
|------|------|------|
| **主模型** | TFNO3dNet (FNO) | 3D 傅里叶神经算子，网格分辨率不变性 |
| **备选模型** | 3D CNN (U-Net) | 快速原型验证 |
| **几何表示** | SDF (Signed Distance Function) | 紧凑隐式几何，类似 DeepCFD |
| **输出** | (u, v, w, p) | 三维速度场 + 压力场 |

```
sedan_aero_ai/
├── configs/sedan_fno.yaml    # 模型/物理/训练全参数配置
├── data/
│   ├── generate.py           # 轿车几何 SDF + 合成流场生成
│   └── __init__.py
├── models/
│   ├── cfd_surrogate.py      # TFNO3dNet + SpectralConv3d + 3D CNN
│   └── __init__.py
├── utils/
│   ├── helpers.py            # 指标/checkpoint/速度对比工具
│   └── __init__.py
├── train.py                  # 训练入口
├── infer.py                  # 推理 + 速度 benchmark
├── requirements.txt
└── README.md
```

---

## 运行环境

### 硬件要求

| 模式 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 训练（CPU） | 8 GB RAM | 16 GB+ RAM |
| 训练（GPU） | 单卡 8 GB 显存 | 单卡 16 GB+ 显存（如 V100/A100） |
| 推理 | 2 GB RAM | — |

### 系统环境

- **操作系统**: macOS 14+ / Linux (Ubuntu 20.04+) / Windows 11
- **架构**: x86_64 / arm64 (Apple Silicon)
- **Python**: 3.10 ~ 3.13
- **已验证环境**:

  | 项 | 版本 |
  |---|------|
  | OS | Darwin 25.5.0 arm64 (macOS Sequoia, Apple Silicon) |
  | Python | 3.13.12 |
  | PaddlePaddle | 3.2.2 (CPU) |

### 依赖安装

```bash
# 基础依赖
pip install -r requirements.txt

# CPU 版 PaddlePaddle（macOS / Linux 通用）
pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 版 PaddlePaddle（需 CUDA 11.8+）
# pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### 依赖清单 (`requirements.txt`)

```
paddlepaddle>=3.0.0
numpy>=1.24.0
pyyaml>=6.0
```

---

## 快速开始

### 训练

```bash
# 使用合成数据快速验证（默认）
python train.py --model fno --epochs 200 --batch-size 4

# 使用 3D CNN 作为替代方案
python train.py --model cnn --epochs 200

# GPU 训练
python train.py --model fno --epochs 500 --batch-size 8 --device gpu
```

### 推理

```bash
# 加载训练好的模型进行推理 + 速度 benchmark
python infer.py --model fno --model-path ./outputs/final_model.pdparams --benchmark

# 多工况预测
python infer.py --model fno --model-path <path> --u-inf 25 30 35

# 单独评估已训练模型（复用训练时生成的验证数据）
python train.py --model fno --eval --output-dir ./outputs
```

---

## 配置说明

核心配置项见 `configs/sedan_fno.yaml`:

| 配置段 | 关键参数 | 含义 |
|--------|---------|------|
| `MODEL` | `n_modes_*`, `hidden_channels`, `n_layers` | FNO 模型结构 |
| `PHYSICS` | `Re`, `U_inf`, `rho`, `nu`, `L_ref`, `A_ref` | 物理参数 |
| `DOMAIN` | `bbox`, `nx/ny/nz`, `car_bbox` | 计算域与网格 |
| `TRAIN` | `epochs`, `batch_size`, `learning_rate` | 训练超参 |
| `DATA` | `use_synthetic`, `train_samples`, `data_dir` | 数据配置 |

---

## 速度对比

| 方法 | 单次推理/仿真时间 | 加速比 |
|------|------------------|--------|
| AI Surrogate (CPU) | ~1 ms | — |
| RANS CFD (OpenFOAM) | ~30 分钟 | ~10^6x |
| LES CFD | ~48 小时 | ~10^8x |

> 当前 benchmark 基于合成数据在 16x8x4 粗网格上测试。接入真实 CFD 数据集后需重新评估物理精度。

---

## 接入真实 CFD 数据

1. 将 `configs/sedan_fno.yaml` 中 `DATA.use_synthetic` 设为 `false`
2. 指定 `DATA.data_dir` 指向 CFD 结果目录
3. 数据格式: 每份样本包含 `.npz` 文件，内含 `sdf`、`u`、`v`、`w`、`p` 字段
4. 推荐的 CFD 工具链: OpenFOAM / SU2 → RANS 计算 → 导出网格速度场/压力场

## License

内部研究项目。
