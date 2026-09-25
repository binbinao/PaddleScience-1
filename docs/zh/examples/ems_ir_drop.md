# EMS 电源平面 DC IR-Drop 仿真

<a href="https://aistudio.baidu.com/" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    python ems_ir_drop.py
    ```

=== "模型评估命令"

    ``` sh
    python ems_ir_drop.py mode=eval EVAL.pretrained_model_path=None
    ```

## 1. 背景简介

发动机管理系统（Engine Management System, EMS）是整车电子控制的核心部件之一，其 ECU（Electronic Control Unit）负责喷油、点火、怠速控制、排放监控等实时任务。ECU 内部的 5V 电源平面为 MCU、点火驱动器、传感器信号调理、CAN 收发器、EEPROM 等负载供电，这些负载在喷油/点火脉冲等工作场景下呈现数十至数百毫安的动态电流。

电源平面的 DC IR-drop（直流压降）直接决定负载端的供电质量：压降过大会导致 MCU 复位阈值漂移、传感器信号调理精度下降、CAN 通讯电平裕量收窄。传统电源完整性（Power Integrity, PI）分析依赖商业 EDA 工具的有限元/有限差分求解器，在早期布局迭代中难以快速反馈。

PINN 替代仿真的价值在于：一次训练后，任意坐标处的电压可即时推断，可在布局迭代的参数扫描中提供快速反馈，并可封装进部署管线（`deploy/python_infer`）做在线估算。

## 2. 问题定义

### 2.1 物理模型

稳态下，铜箔电源平面内的电势 $u(x, y)$ 满足 Laplace 方程（电流源以边界条件形式给出）：

$$
\nabla \cdot (\sigma_s \nabla u) = 0, \quad \text{in } \Omega,
$$

其中 $\sigma_s = \sigma t$ 为铜箔的薄层电导（电导率乘以箔厚）。边界条件分三类：

1. VRM（DC-DC 变换器焊盘）：恒压源，$u = 0$（Dirichlet，参考地取 VRM 焊盘电势）；
2. 负载焊盘：注入电流 $I_k$（Neumann），$-\sigma_s \partial u/\partial n = I_k/(2\pi r_k) $ 沿焊盘环均匀注入；
3. 平面其余边界（板边、隔离孔壁、安装孔壁）：绝缘，$\partial u/\partial n = 0$。

几何构型取自典型 EMS 主板电源平面：160 mm × 100 mm 双面覆铜板，VRM 位于左侧中部，MCU/点火驱动/传感器/CAN/EEPROM 五个负载焊盘分布于右侧与四角；板面开有 M3 安装孔阵列，其中 x = 70 mm 处的一列安装孔起到隔离作用（真实 EMS PCB 中常用过孔排代替长槽，阻断直通电流路径的同时保持场的光滑性）。

### 2.2 奇性分离

点/小圆盘电流源导致 $u$ 在焊盘附近存在 $\ln(1/r)$ 奇性，直接用 PINN 拟合会因梯度爆炸而失效（冒烟实验已证实：网络塌缩到 $u \equiv 0$）。采用奇性分离：

$$
u = v + u_s, \quad u_s = \sum_k a_k \ln\frac{1}{r_k}, \quad a_k = \frac{I_k / I_{tot}}{2\pi},
$$

网络只学光滑部分 $v$，其约束为：

- 内部：$\Delta v = 0$；
- VRM 环：$v = -u_s$（Dirichlet）；
- 负载焊盘环：$\partial v/\partial n = 0$（$u_s$ 已精确携带注入通量）；
- 绝缘壁：$\partial v/\partial n = -\nabla u_s \cdot \mathbf{n}$。

### 2.3 自研 FDM 参考解与锚点弱监督

本案例不依赖任何外部数据，参考解由脚本内置的有限体积 FDM 求解器（`solve_ir_drop_fd`，纯 numpy/scipy 稀疏矩阵）在阶梯逼近的铜箔网格上计算，并通过 VRM 吸收电流守恒校验（= 1.000000）。它承担两个角色：

1. **验证基准**：验证器在全部铜箔节点上对比 $u = v + u_s$ 与 FDM 参考场（L2Rel / MaxAE），训练结束后输出焊盘压降/等效电阻对照表；
2. **稀疏锚点弱监督**：纯边界驱动的 PINN 训练在此类 Neumann 主导问题上会停滞在 $v \approx 0$ 的欺骗解——边界通量信息通过损失的传播太弱，无法固定远场幅度（PINN 领域已知的梯度病理，见参考资料）。为此从 FDM 铜网格中抽取 2000 个内部锚点（约占 FDM 节点的 12%）作为稀疏 Dirichlet 监督，锚定远场幅度；PDE/壁面约束保证锚点之间场的物理一致性。这一“物理引导插值”设定如实标注了监督来源，且完全自洽（无外部数据）。

## 3. 问题求解

脚本遵循标准 Hydra 模式（`train` / `eval` / `export` / `infer` 四模式），下面按代码段落讲解关键步骤。

### 3.1 模型构建

一个普通 MLP 将坐标 $(x, y)$ 映射到光滑部分 $v$：

``` yaml linenums="72"
--8<--
examples/ems_ir_drop/conf/ems_ir_drop.yaml:72:77
--8<--
```

``` py linenums="616"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:616:617
--8<--
```

### 3.2 计算域构建

电源平面由 CSG 图元组装：板矩形减去安装孔并集（含 x=0.70 处的隔离孔列）、再减去 VRM 与负载焊盘圆盘：

``` py linenums="311"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:311:347
--8<--
```

### 3.3 约束构建

`build_constraints` 构建五组约束：

- `EQ`：$v$ 的内部 Laplace 残差（`InteriorConstraint`）；
- `VRM`：VRM 环 Dirichlet $v = -u_s$（`BoundaryConstraint`）；
- `PAD_<名称>`：每个负载焊盘的零通量环约束（每个焊盘独立约束、损失键唯一 `flux_<名称>`）；
- `EDGE` / `SLOT`（可选）/ `HOLE`：绝缘壁法向通量约束，label 为闭式 $-\nabla u_s \cdot \mathbf{n}$；
- `ANCHOR`：FDM 稀疏锚点监督（`SupervisedConstraint` + `NamedArrayDataset`），权重由 `TRAIN.anchor_weight` 控制。

``` py linenums="398"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:398:520
--8<--
```

### 3.4 FDM 参考求解器与验证器

FDM 求解器构建阶梯铜箔掩码、组装含焊盘电流源与 VRM 汇的稀疏电导矩阵、以 `scipy.sparse.linalg.spsolve` 直接求解；守恒性通过对比 VRM 吸收电流与总注入电流校验。

验证器排除 VRM/焊盘圆盘内部的网格节点（焊盘圆心处解析 $u_s$ 奇性），在其余铜节点上对比 $u = v + u_s$ 与 FDM 参考解：

``` py linenums="522"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:522:578
--8<--
```

### 3.5 模型训练

Adam 优化器固定学习率训练 800 epochs，每 200 epochs 对照 FDM 参考评估一次：

``` py linenums="616"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:616:700
--8<--
```

训练结束后输出焊盘压降对照表（PINN vs FDM 压降与等效电阻）：

``` py linenums="580"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:580:614
--8<--
```

## 4. 完整代码

``` py linenums="1" title="ems_ir_drop.py"
--8<--
examples/ems_ir_drop/ems_ir_drop.py
--8<--
```

## 5. 结果展示

默认配置训练 800 epochs 后（CPU，约 22 分钟），验证集（FDM_N=200 全铜节点）指标：

| 指标 | 数值 |
| :-- | :-- |
| L2 相对误差（u 场） | 0.1048 |
| 最大绝对误差（u 场，归一化） | 0.478 |

各负载焊盘的压降与等效电阻对照（物理单位）：

| 焊盘 | 电流 [A] | PINN 压降 [mV] | FDM 压降 [mV] | PINN 电阻 [mΩ] | FDM 电阻 [mΩ] | 压降偏差 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| MCU | 0.50 | 0.4263 | 0.4625 | 0.853 | 0.925 | -7.8% |
| IGN | 0.60 | 0.4323 | 0.4688 | 0.721 | 0.781 | -7.8% |
| SENS | 0.15 | 0.3082 | 0.3254 | 2.054 | 2.169 | -5.3% |
| CAN | 0.08 | 0.2222 | 0.2133 | 2.778 | 2.666 | +4.2% |
| EEP | 0.05 | 0.4146 | 0.4452 | 8.292 | 8.903 | -6.9% |

五个焊盘压降偏差全部在 ±8% 以内，满足布局早期快速估算的精度需求。压降分布的物理合理性亦得到复现：右侧（跨过隔离孔列的）MCU/EEP 负载压降高于左侧近 VRM 的 CAN 收发器，EEP 因电流最小（0.05 A）而等效电阻最大。

训练与评估结束后，`visual/` 目录输出全平面 $u$ 场的 vtu 文件（归一化范围 [0, 1.30]，对应物理 0 ~ 0.44 mV），可在 ParaView 中查看；`mode=export` 导出静态推理模型，`mode=infer` 通过 `deploy/python_infer` 加载导出模型批量推断。

## 6. 参考资料

参考文献：

- Wang, S., Teng, Y., Perdikaris, P. "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks", SIAM J. Sci. Comput., 2021.
