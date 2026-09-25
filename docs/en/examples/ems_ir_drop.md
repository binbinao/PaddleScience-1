# EMS Power Plane DC IR-Drop Simulation

<a href="https://aistudio.baidu.com/" class="md-button md-button--primary" style>Quick experience on AI Studio</a>

=== "Training command"

    ``` sh
    python ems_ir_drop.py
    ```

=== "Evaluation command"

    ``` sh
    python ems_ir_drop.py mode=eval EVAL.pretrained_model_path=None
    ```

## 1. Background

The Engine Management System (EMS) is one of the core electronic control units in modern vehicles: its ECU runs fuel injection, ignition, idle-speed control, and emission monitoring in real time. Inside the ECU, a 5 V power plane feeds the MCU, ignition driver, sensor signal-conditioning cluster, CAN transceiver, and EEPROM. During injection/ignition pulses these loads draw tens to hundreds of milliamperes.

The DC IR-drop of the power plane directly determines supply quality at the loads: excessive drop shifts the MCU reset threshold, degrades sensor conditioning accuracy, and narrows CAN level margin. Conventional power-integrity (PI) analysis relies on finite-element / finite-difference solvers inside commercial EDA tools, which are slow to iterate during early layout.

The value of a PINN surrogate: after a single training run, the voltage at any coordinate is instantly available, enabling fast feedback sweeps during layout iteration, and the trained model can be deployed through `deploy/python_infer` for online estimation.

## 2. Problem Definition

### 2.1 Physical model

At steady state the copper power-plane potential $u(x, y)$ satisfies the Laplace equation (current sources enter as boundary conditions):

$$
\nabla \cdot (\sigma_s \nabla u) = 0, \quad \text{in } \Omega,
$$

where $\sigma_s = \sigma t$ is the sheet conductance (conductivity times copper thickness). Boundary conditions come in three kinds:

1. VRM (DC-DC buck converter pad): constant-voltage source, $u = 0$ (Dirichlet; the reference ground is taken at the VRM pad);
2. Load pads: injected current $I_k$ (Neumann), $-\sigma_s \partial u/\partial n = I_k/(2\pi r_k)$, injected uniformly along the pad ring;
3. All remaining boundaries (board edges, mounting-hole walls): insulated, $\partial u/\partial n = 0$.

### 2.2 Singularity split

Point / small-disk current sources produce an $\ln(1/r)$ singularity of $u$ near the pads. Fitting it directly makes PINN training collapse (verified by a smoke experiment: the network degenerates to $u \equiv 0$). We therefore split

$$
u = v + u_s, \quad u_s = \sum_k a_k \ln\frac{1}{r_k}, \quad a_k = \frac{I_k / I_{tot}}{2\pi},
$$

and the network only learns the smooth part $v$, governed by:

- interior: $\Delta v = 0$;
- VRM ring: $v = -u_s$ (Dirichlet);
- load-pad rings: $\partial v/\partial n = 0$ ($u_s$ already carries the exact injected flux);
- insulated walls: $\partial v/\partial n = -\nabla u_s \cdot \mathbf{n}$.

### 2.3 FDM reference and anchor supervision

A self-developed finite-volume FDM solver (`solve_ir_drop_fd`, pure numpy/scipy sparse) computes the reference solution on the staircase copper grid. It serves two purposes:

1. **Validation**: the validator compares $u = v + u_s$ against the FDM field everywhere on the copper (L2Rel / MaxAE), and the pad-voltage table contrasts PINN and FDM effective resistances.
2. **Sparse anchor supervision**: pure boundary-driven PINN training on this Neumann-heavy problem stalls in a deceptive $v \approx 0$ solution — the boundary-flux information propagates too weakly through the loss to set the far-field amplitude (a known PINN pathology). A sparse set of interior anchor values (2000 points, about 12% of the FDM copper nodes) pins that amplitude; the PDE/wall constraints keep the field physically consistent between anchors. This physics-guided-interpolation setup is honest about what is supervised and remains fully self-contained (no external data).

## 3. Solving with PaddleScience

The script follows the standard Hydra pattern. Key steps map to code sections of `ems_ir_drop.py`:

### 3.1 Model

A plain MLP maps $(x, y)$ to the smooth part $v$:

``` py linenums="72"
--8<--
examples/ems_ir_drop/conf/ems_ir_drop.yaml:72:77
--8<--
```

``` py linenums="616"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:616:617
--8<--
```

### 3.2 Geometry

The plane is assembled from CSG primitives: the board rectangle minus the mounting-hole union (including the isolation hole row at x=0.70 replacing a long slot, as on real EMS PCBs), minus the VRM and load-pad disks:

``` py linenums="311"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:311:347
--8<--
```

### 3.3 Constraints

Five constraint groups are built in `build_constraints`:

- `EQ`: interior Laplace residual on $v$ (`InteriorConstraint`);
- `VRM`: Dirichlet ring $v = -u_s$ (`BoundaryConstraint`);
- `PAD_<name>`: zero-flux rings around each load pad (one constraint per pad, unique loss keys `flux_<name>`);
- `EDGE` / `SLOT`(optional) / `HOLE`: insulated-wall flux with the closed-form label $-\nabla u_s \cdot \mathbf{n}$;
- `ANCHOR`: sparse FDM anchor supervision (`SupervisedConstraint` over a `NamedArrayDataset`), weighted by `TRAIN.anchor_weight`.

``` py linenums="398"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:398:520
--8<--
```

### 3.4 FDM reference solver and validator

The finite-volume FDM solver (`solve_ir_drop_fd`) builds the staircase copper mask, assembles the sparse conductance matrix with pad current sources and the VRM sink, and solves with `scipy.sparse.linalg.spsolve`. Conservation is verified by comparing the absorbed VRM current against the total injected current.

The validator excludes grid nodes inside the VRM/pad disks (where the analytic $u_s$ is singular at pad centers) and compares $u = v + u_s$ against the FDM reference:

``` py linenums="522"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:522:578
--8<--
```

### 3.5 Training

Adam with a fixed learning rate trains for 800 epochs; evaluation against the FDM reference runs every 200 epochs:

``` py linenums="616"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:616:700
--8<--
```

After training, the pad-voltage table reports PINN vs FDM drop and effective resistance per load:

``` py linenums="580"
--8<--
examples/ems_ir_drop/ems_ir_drop.py:580:614
--8<--
```

## 4. Complete code

``` py linenums="1" title="ems_ir_drop.py"
--8<--
examples/ems_ir_drop/ems_ir_drop.py
--8<--
```

## 5. Results

After 800 epochs of training with the default configuration (CPU, ~22 minutes), the validator (all copper nodes at FDM_N=200) reports:

| Metric | Value |
| :-- | :-- |
| L2 relative error (u field) | 0.1048 |
| Max absolute error (u field, normalized) | 0.478 |

Per-pad drop and effective resistance (physical units):

| Pad | Current [A] | PINN drop [mV] | FDM drop [mV] | PINN R [mΩ] | FDM R [mΩ] | Drop deviation |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| MCU | 0.50 | 0.4263 | 0.4625 | 0.853 | 0.925 | -7.8% |
| IGN | 0.60 | 0.4323 | 0.4688 | 0.721 | 0.781 | -7.8% |
| SENS | 0.15 | 0.3082 | 0.3254 | 2.054 | 2.169 | -5.3% |
| CAN | 0.08 | 0.2222 | 0.2133 | 2.778 | 2.666 | +4.2% |
| EEP | 0.05 | 0.4146 | 0.4452 | 8.292 | 8.903 | -6.9% |

All five pad drops deviate by less than ±8% from the FDM reference, sufficient for early-layout fast estimation. The physical trend is reproduced as well: pads on the far side of the isolation hole row (MCU/EEP) see larger drops than the CAN transceiver near the VRM, and the EEPROM shows the largest effective resistance due to its small current (0.05 A).

After training and evaluation, the `visual/` directory contains vtu files of the full-plane $u$ field (normalized range [0, 1.30], i.e. 0 ~ 0.44 mV physical) viewable in ParaView; `mode=export` dumps a static inference model, and `mode=infer` runs batch prediction through `deploy/python_infer`.

## 6. References

- Wang, S., Teng, Y., Perdikaris, P. "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks", SIAM J. Sci. Comput., 2021.
