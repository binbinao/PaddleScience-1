# Sedan Aero AI

This case builds a sedan external-aerodynamics AI surrogate with a 3D Fourier Neural Operator (FNO). Taking the signed distance function (SDF) of the car geometry as input, it directly predicts the surrounding flow field $(u, v, w, p)$, achieving orders-of-magnitude speedup over conventional RANS CFD simulations.

The case is a standalone implementation (argparse + YAML config) that does not rely on PaddleScience's Hydra config system; it runs with nothing but PaddlePaddle installed.

=== "Model Training Command"

    ``` sh
    cd examples/operator_learning/sedan_aero_ai
    # FNO architecture (default); synthetic data is generated automatically
    python train.py --model fno
    # 3D CNN architecture
    python train.py --model cnn
    # Override training hyperparameters
    python train.py --model fno --epochs 500 --batch-size 4 --lr 5e-4
    ```

=== "Model Evaluation Command"

    ``` sh
    # Reuse the validation data generated during training
    python train.py --model fno --eval --output-dir ./outputs
    # Or point at a specific checkpoint
    python train.py --model fno --eval --pretrained ./outputs/final_model.pdparams --output-dir ./outputs
    ```

=== "Model Inference Command"

    ``` sh
    # Load a trained model, run inference + speed benchmark
    python infer.py --model fno --model-path ./outputs/final_model.pdparams --benchmark
    # Multi-condition prediction (different inlet velocities)
    python infer.py --model fno --model-path ./outputs/final_model.pdparams --u-inf 25 30 35
    ```

Note: after training completes, an inference benchmark runs automatically; results land in `outputs/benchmark.json` and `outputs/prediction.npz`.

## 1. Background Introduction

Sedan aerodynamics development relies on massive CFD campaigns: one RANS run costs ~30 minutes on 32 cores, and LES tens of hours, which makes large-scale design-space sweeps over shapes impractical. Neural operator learning offers an alternative: conditioned on a geometry representation, learn the "geometry → flow field" map directly and compress inference cost to milliseconds.

[FNO](https://export.arxiv.org/pdf/2010.08895.pdf) applies learnable complex linear transforms to low-frequency modes in the Fourier domain, combining a global receptive field with resolution invariance — a good backbone for flow-field surrogates.

The current implementation uses **synthetic flow data** (potential-flow + empirical wake model) to exercise the pipeline end to end; see [Section 3.8](#38-plugging-in-real-cfd-data) for switching to real CFD data.

## 2. Problem Definition

Given a parametrized sedan geometry $\Omega_{car}$ and inflow conditions (velocity $U_\infty$, Reynolds number $Re$), solve the 3D flow field around it:

$$
\begin{equation}
\mathcal{G}: \big(\text{SDF}(\mathbf{x}),\ \mathbf{x}\big) \mapsto \big(u,\ v,\ w,\ p\big)(\mathbf{x}), \quad \mathbf{x} \in \Omega \setminus \Omega_{car}
\end{equation}
$$

The geometry is encoded as a signed distance function (negative inside the car, positive outside) over a rectangular bounding-box domain (default 40m × 16m × 8m, car centered). Training minimizes the mean squared error between predicted and reference flow fields (CFD or synthetic).

Default physical parameters: $Re = 2\times10^6$, $U_\infty = 30\,\mathrm{m/s}$ (~108 km/h cruise), car length $L_{ref} = 4.5\,\mathrm{m}$, frontal area $A_{ref} = 2.2\,\mathrm{m^2}$.

## 3. Problem Solving

The case is self-contained; components live in `examples/operator_learning/sedan_aero_ai/` and are imported from sibling `models/`, `data/`, `utils/` packages via `sys.path.insert`.

### 3.1 Dataset Construction

`data/generate.py` provides two building blocks:

1. **Parametric sedan geometry** (`SedanGeometry`): builds the SDF from a piecewise top profile (hood → windshield → roof → rear window → trunk) plus wheel ellipsoids; `get_sdf` evaluates it on the bounding-box grid.

2. **Synthetic flow generation** (`FlowFieldGenerator`): produces physically structured flow fields from a simplified potential-flow + wake-deficit model (near-body slowdown, wake velocity deficit, upwash/downwash, wake pressure depression). Note: synthetic data exercises the pipeline only — it is not physically accurate.

``` py linenums="296"
--8<--
examples/operator_learning/sedan_aero_ai/data/generate.py:296:303
--8<--
```

Training/val samples are generated per `DATA.train_samples` / `val_samples` with inlet velocity sampled randomly from a given range; `create_fno_dataset` then stacks them into FNO tensor format:

- Input: `(N, 1, D, H, W)` — single-channel SDF voxels
- Output: `(N, 4, D, H, W)` — four-channel flow field $(u, v, w, p)$

### 3.2 Model Construction

The primary model `SedanAeroFNO` (`models/cfd_surrogate.py`) is a self-contained 3D FNO implementation (with PaddleScience installed you may also use `ppsci.arch.TFNO3dNet` directly). Each FNO layer combines:

$$
x_{l+1} = \sigma\big(\mathcal{F}^{-1}(R_l \cdot \mathcal{F}(x_l)) + W_l x_l\big)
$$

i.e. a learnable complex transform on low-frequency Fourier modes plus a 1×1×1 convolutional skip path:

``` py linenums="169"
--8<--
examples/operator_learning/sedan_aero_ai/models/cfd_surrogate.py:169:205
--8<--
```

The alternative `SedanAeroCNN` is a 3D U-Net with skip connections, suited to quick prototyping. Select via `--model {fno,cnn}`; architecture parameters (mode counts, hidden channels, layers) live in the `MODEL` block of `configs/sedan_fno.yaml`.

### 3.3 Hyperparameters and Optimizer

Training defaults: 500 epochs, batch size 4, MSE loss; Adam optimizer (initial lr 1e-3, weight decay 1e-4) with a `StepDecay` schedule (0.5× every 100 epochs):

``` py linenums="227"
--8<--
examples/operator_learning/sedan_aero_ai/train.py:227:237
--8<--
```

Every hyperparameter can be overridden on the CLI (`--epochs/--batch-size/--lr`); see `python train.py -h`.

### 3.4 Training Procedure

`train.py` runs: generate (or load) dataset → build model and optimizer → per-epoch training, evaluating on the validation set every `TRAIN.eval_freq` epochs (tracking the best) and saving checkpoints every `TRAIN.save_freq` epochs (`CheckpointManager` keeps at most 5). After training it saves `final_model.pdparams`, `metrics.json`, and runs one inference benchmark automatically.

### 3.5 Inference and Aerodynamic Coefficients

`infer.py` loads trained weights and supports:

- **Single / multi-condition prediction** (`--u-inf 25 30 35`): generates a test geometry per inlet velocity and predicts the flow field;
- **Speed benchmark** (`--benchmark`): 3 warmup passes, then timing averaged over 10 runs; reports throughput and speedup vs RANS (30 min) / LES (48 h);
- **Aerodynamic coefficient estimation**: numerically integrates the predicted pressure field over the body surface (where SDF ≈ 0) to estimate $C_d$, $C_l$.

Outputs land in `--output-dir` (default `./inference_results`): `prediction.npz` (flow field + coordinates + SDF), `benchmark.json`, `aerodynamics.json`.

### 3.6 Results

Measured on a synthetic-data coarse grid (16×8×4) on CPU: ~9 ms per FNO inference (~108 samples/s), ~1.6 ms for CNN; speedup vs RANS CFD (30 minutes) reaches the $10^5$ range. These numbers reflect pipeline speed only — physical accuracy must be re-evaluated on real CFD data.

### 3.7 Layout and Configuration

```
sedan_aero_ai/
├── configs/sedan_fno.yaml    # model/physics/training configuration
├── data/generate.py          # sedan geometry SDF + synthetic flow generation
├── models/cfd_surrogate.py   # SedanAeroFNO + SpectralConv3d + SedanAeroCNN
├── utils/helpers.py          # metrics/checkpoint/benchmark utilities
├── train.py                  # training / evaluation entry
├── infer.py                  # inference + benchmark entry
└── requirements.txt          # paddlepaddle, numpy, pyyaml
```

Config blocks: `MODEL` (FNO architecture), `PHYSICS` ($Re$, $U_\infty$, $\rho$, $\nu$, reference areas), `DOMAIN` (bounding box and grid resolution), `TRAIN` (epochs/batch/lr schedule), `DATA` (synthetic switch, sample counts, real-data dir).

### 3.8 Plugging in Real CFD Data

1. Run RANS simulations with OpenFOAM / SU2 etc. and export the flow fields per geometry;
2. Set `DATA.use_synthetic: false` in `configs/sedan_fno.yaml`;
3. Prepare `train.npz` / `val.npz` (fields: `input` of shape `(N, 1, D, H, W)` holding SDF, `output` of shape `(N, 4, D, H, W)` holding $(u,v,w,p)$) inside `DATA.data_dir`.

## 4. Complete Code

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

## 5. References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://export.arxiv.org/pdf/2010.08895.pdf)
- [Neural Operator: Learning Maps Between Function Spaces](https://export.arxiv.org/pdf/2108.08481.pdf)
- [TFNO3dNet — PaddleScience API](https://paddlescience-docs.readthedocs.io/zh-cn/latest/api/arch/)
