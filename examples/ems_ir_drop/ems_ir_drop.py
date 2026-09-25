# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PINN surrogate for DC IR-drop on an EMS engine-management ECU power plane.

Physical setup (geometry scaled to [0, 1.6] x [0, 1]; 1 unit = 100 mm, i.e. a
160 mm x 100 mm board):
- A 2oz copper 5V power plane with a vertical anti-routing slot leaving only a
  15 mm copper bridge at the bottom, four M3 mounting holes, one VRM pad
  (DC-DC buck regulator output, Dirichlet u=0) and five load pads injecting
  DC current: MCU, ignition driver, sensor cluster, CAN transceiver and
  EEPROM.
- The normalized potential drop u = (V_vrm - phi) / U0 (U0 = I_tot / sigma_s)
  satisfies Laplace's equation on the copper with u=0 on the VRM pad ring,
  prescribed inward normal derivative on each load pad ring and zero normal
  derivative elsewhere.

Numerics: point-pad Neumann data carries a logarithmic singularity which a
plain tanh MLP cannot resolve, so the solution is split into a closed-form
singular part and a smooth network part:

    u(x, y) = v(x, y) + u_s(x, y),
    u_s(x, y) = sum_k a_k * ln(1 / r_k),  a_k = (I_k / I_tot) / (2 pi),

whose outward flux on pad k's ring equals the prescribed injection. The
network only learns the smooth remainder v, which satisfies:

- Laplace's equation on the copper interior,
- v = -u_s on the VRM pad ring (Dirichlet),
- dv/dn = 0 on every load pad ring (the singularity carries the flux),
- dv/dn = -grad(u_s) . n on insulated walls (board edge, slot, holes),

where n is the boundary normal sampled by ppsci (pointing out of the sampled
geometry). A self-contained node-centered finite-volume FDM reference solver
on a Cartesian grid provides validation labels and effective pad resistances
R_eff = V_pad / I_k.
"""

import os.path as osp

import hydra
import numpy as np
import paddle
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base
from ppsci.utils import logger


class LaplaceV(base.PDE):
    r"""Laplace equation on the smooth part v of the potential split.

    $$
    \nabla^2 v = 0
    $$
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        invars = self.create_symbols("x y z")[:dim]
        v = self.create_function("v", invars)
        laplace_v = 0
        for invar in invars:
            laplace_v += v.diff(invar, 2)
        self.add_equation("laplace_v", laplace_v)


# ---------------------------------------------------------------------------
# analytical singularity term u_s and its gradient
# ---------------------------------------------------------------------------
def _u_singularity_terms(cfg):
    """Coefficients of the analytical logarithmic singularity term.

    u_s(x, y) = sum_k a_k * ln(1 / r_k) with a_k = (I_k / I_tot) / (2 pi)
    carries the pad injection flux analytically: on pad k's ring,
    du_s/dn_copper = a_k / r_k = (I_k/I_tot) / (2 pi r_k), exactly the
    prescribed inward flux. u_s is harmonic away from the pads.

    Args:
        cfg (DictConfig): Runtime config.

    Returns:
        List of (cx, cy, r, a_k) tuples for all load pads.
    """
    i_total = sum(p[4] for p in cfg.PADS)
    return [(p[1], p[2], p[3], (p[4] / i_total) / (2.0 * np.pi)) for p in cfg.PADS]


def _u_s(x, y, terms):
    """Evaluate the singularity term u_s and its gradient (numpy).

    Args:
        x, y (np.ndarray): Coordinates of shape [N, 1].
        terms (List): Output of `_u_singularity_terms`.

    Returns:
        Tuple (u_s, du_s/dx, du_s/dy) as numpy arrays of x's shape.
    """
    x = np.asarray(x, "float64").reshape(-1, 1)
    y = np.asarray(y, "float64").reshape(-1, 1)
    us = np.zeros_like(x)
    dus_dx = np.zeros_like(x)
    dus_dy = np.zeros_like(x)
    for cx, cy, _, a in terms:
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        us += a * np.log(1.0 / np.sqrt(r2))
        dus_dx += -a * dx / r2
        dus_dy += -a * dy / r2
    return us, dus_dx, dus_dy


def _u_s_paddle(out, terms):
    """Paddle version of `_u_s` for use inside output_expr callables.

    Args:
        out (Dict[str, paddle.Tensor]): Input dict with keys "x", "y".
        terms (List): Output of `_u_singularity_terms`.

    Returns:
        Tuple (u_s, du_s/dx, du_s/dy) as paddle tensors.
    """
    x, y = out["x"], out["y"]
    us = paddle.zeros_like(x)
    dus_dx = paddle.zeros_like(x)
    dus_dy = paddle.zeros_like(x)
    for cx, cy, _, a in terms:
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        us += a * paddle.log(r2) * (-0.5)
        dus_dx += -a * dx / r2
        dus_dy += -a * dy / r2
    return us, dus_dx, dus_dy


# ---------------------------------------------------------------------------
# FDM reference solver (node-centered finite volume, staircase holes)
# ---------------------------------------------------------------------------
def solve_ir_drop_fd(
    board,
    slot,
    vrm,
    pads,
    holes,
    hole_r: float,
    n: int,
):
    """Solve the normalized IR-drop field with a finite-volume FDM scheme.

    The board is discretized on a uniform Cartesian grid (n cells per unit
    length). Nodes inside the slot, the mounting holes, the VRM pad and the
    load pads are excluded; the VRM ring is held at u=0 (Dirichlet), each load
    pad injects its normalized current uniformly over its staircase boundary
    faces, and all remaining boundaries are insulated.

    Args:
        board (Tuple[float, float]): Board extent (Lx, Ly) in normalized units.
        slot (Tuple[float, float, float, float]): Slot rect (xmin, xmax, ymin,
            ymax); the slot reaches the top edge so ymax is only clipped.
        vrm (Tuple[str, float, float, float]): VRM (name, cx, cy, r).
        pads (List[Tuple[str, float, float, float, float]]): Load pads as
            (name, cx, cy, r, current in Ampere).
        holes (List[Tuple[float, float]]): Mounting hole centers.
        hole_r (float): Mounting hole radius.
        n (int): Grid resolution (cells per unit length).

    Returns:
        Dict with grid arrays X, Y, solution U (NaN outside copper), the
        copper node mask, VRM absorbed current (conservation check) and grid
        spacing h.
    """
    Lx, Ly = board
    i_total = sum(p[4] for p in pads)
    nx, ny = round(Lx * n) + 1, round(Ly * n) + 1
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if slot is not None:
        in_hole = (X > slot[0]) & (X < slot[1]) & (Y > slot[2]) & (Y < slot[3])
    else:
        in_hole = np.zeros(X.shape, dtype=bool)
    for hx, hy in holes:
        in_hole |= (X - hx) ** 2 + (Y - hy) ** 2 < hole_r**2
    is_sink = (X - vrm[1]) ** 2 + (Y - vrm[2]) ** 2 < vrm[3] ** 2
    copper = ~in_hole & ~is_sink
    K = int(copper.sum())
    idx = np.full(X.shape, -1, np.int64)
    idx[copper] = np.arange(K)
    src_mask = [(X - p[1]) ** 2 + (Y - p[2]) ** 2 < p[3] ** 2 for p in pads]

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def _shift(d):
        dx, dy = d
        return (
            (
                slice(max(0, -dx), nx - max(0, dx)),
                slice(max(0, -dy), ny - max(0, dy)),
            ),
            (
                slice(max(0, dx), nx - max(0, -dx)),
                slice(max(0, dy), ny - max(0, -dy)),
            ),
        )

    # count staircase boundary faces of each load pad
    n_faces = [0] * len(pads)
    for d in dirs:
        ps, qs = _shift(d)
        for k, sm in enumerate(src_mask):
            n_faces[k] += int((copper[ps] & sm[qs]).sum())
    face_flux = [(p[4] / i_total) / n_faces[k] for k, p in enumerate(pads)]

    rows, cols, vals = [], [], []
    diag = np.zeros(K)
    rhs = np.zeros(K)
    for d in dirs:
        ps, qs = _shift(d)
        p_cu, q_cu, q_sk = copper[ps], copper[qs], is_sink[qs]
        # copper-copper faces contribute -u_neighbor to the flux balance
        m = p_cu & q_cu
        ip, iq = idx[ps][m], idx[qs][m]
        rows.append(ip)
        cols.append(iq)
        vals.append(np.full(ip.size, -1.0))
        diag[ip] += 1.0
        # copper-VRM faces: Dirichlet u=0, contributes to the diagonal only
        m = p_cu & q_sk
        diag[idx[ps][m]] += 1.0
        # copper-pad faces: uniform injection of the pad's face flux
        for k, sm in enumerate(src_mask):
            m = p_cu & sm[qs]
            rhs[idx[ps][m]] += face_flux[k]
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(K, K),
    ).tocsr() + sp.diags(diag)
    u = spla.spsolve(A, rhs)
    U = np.full(X.shape, np.nan)
    U[copper] = u
    U[is_sink] = 0.0

    # conservation check: total current absorbed by the VRM must equal 1
    vrm_in = 0.0
    for d in dirs:
        ps, qs = _shift(d)
        m = copper[ps] & is_sink[qs]
        vrm_in += float(U[ps][m].sum())
    return {
        "X": X,
        "Y": Y,
        "U": U,
        "copper": copper,
        "vrm_in": vrm_in,
        "h": 1.0 / n,
    }


def fd_eval_pads(fd, vrm, pads):
    """Average ring voltage of every pad over its staircase copper ring.

    Args:
        fd (Dict): Result dict from `solve_ir_drop_fd`.
        vrm (Tuple[str, float, float, float]): VRM pad definition.
        pads (List[Tuple]): Load pad definitions.

    Returns:
        Dict mapping pad name to (mean normalized pad voltage, node count).
    """
    X, Y, U, copper, h = (
        fd["X"],
        fd["Y"],
        fd["U"],
        fd["copper"],
        fd["h"],
    )
    out = {}
    for name, cx, cy, r in [(vrm[0], vrm[1], vrm[2], vrm[3])] + [
        (p[0], p[1], p[2], p[3]) for p in pads
    ]:
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        m = copper & (dist >= r - 1e-12) & (dist < r + 1.01 * h)
        out[name] = (float(U[m].mean()), int(m.sum()))
    return out


# ---------------------------------------------------------------------------
# geometry / constraint / validator assembly shared by train and eval
# ---------------------------------------------------------------------------
def build_geometry(cfg):
    """Build the CSG geometries of the power plane.

    Args:
        cfg (DictConfig): Runtime config.

    Returns:
        Dict with the interior geometry (plane minus holes/pads), the
        board-edge rectangle, the optional slot rectangle and the
        mounting-hole union used for boundary constraints, plus the VRM
        and load pad disks.
    """
    rect = ppsci.geometry.Rectangle((0.0, 0.0), (cfg.BOARD[0], cfg.BOARD[1]))
    slot = None
    if cfg.SLOT is not None:
        slot = ppsci.geometry.Rectangle(
            (cfg.SLOT[0], cfg.SLOT[2]), (cfg.SLOT[1], cfg.SLOT[3])
        )
    holes = ppsci.geometry.Disk(cfg.HOLES_COORD[0], cfg.HOLE_R)
    for c in cfg.HOLES_COORD[1:]:
        holes = holes | ppsci.geometry.Disk(c, cfg.HOLE_R)
    vrm_geom = ppsci.geometry.Disk(cfg.VRM[1:3], cfg.VRM[3])
    pad_geoms = {p[0]: ppsci.geometry.Disk((p[1], p[2]), p[3]) for p in cfg.PADS}

    interior = rect - holes - vrm_geom
    if slot is not None:
        interior = interior - slot
    for g in pad_geoms.values():
        interior = interior - g

    return {
        "interior": interior,
        "board_edge": rect,
        "slot_wall": slot,
        "holes_union": holes,
        "vrm": vrm_geom,
        "pads": pad_geoms,
    }


def build_anchor_points(cfg, n_anchor):
    """Sample anchor points from the FDM copper grid for weak supervision.

    Pure boundary-driven PINN training stalls in a deceptive v ~ 0 solution
    (Neumann-type information propagates too weakly through the loss). A
    sparse set of interior anchor values from the self-developed FDM
    reference pins the far-field amplitude and guides the optimizer into
    the basin of the true solution; the remaining PDE/boundary constraints
    keep the field physically consistent between anchors.

    Args:
        cfg (DictConfig): Runtime config.
        n_anchor (int): Number of anchor points (0 disables supervision).

    Returns:
        Tuple of input arrays (x, y) and label array v, or None if disabled.
    """
    if n_anchor <= 0:
        return None
    fd = solve_ir_drop_fd(
        (cfg.BOARD[0], cfg.BOARD[1]),
        cfg.SLOT,
        cfg.VRM,
        cfg.PADS,
        cfg.HOLES_COORD,
        cfg.HOLE_R,
        cfg.FDM_N,
    )
    X, Y, U, copper = fd["X"], fd["Y"], fd["U"], fd["copper"]
    inside_disk = np.zeros_like(copper)
    for _, cx, cy, r_pad, _ in cfg.PADS:
        inside_disk |= (X - cx) ** 2 + (Y - cy) ** 2 <= r_pad**2
    inside_disk |= (X - cfg.VRM[1]) ** 2 + (Y - cfg.VRM[2]) ** 2 <= cfg.VRM[3] ** 2
    valid = copper & ~inside_disk
    idx = np.argwhere(valid)
    rng = np.random.default_rng(cfg.seed)
    sel = rng.choice(len(idx), min(n_anchor, len(idx)), replace=False)
    xs = X[idx[sel][:, 0], idx[sel][:, 1]].reshape(-1, 1)
    ys = Y[idx[sel][:, 0], idx[sel][:, 1]].reshape(-1, 1)
    us, _, _ = _u_s(xs, ys, _u_singularity_terms(cfg))
    vs = (U[idx[sel][:, 0], idx[sel][:, 1]].reshape(-1, 1) - us).astype("float32")
    return {
        "x": xs.astype("float32"),
        "y": ys.astype("float32"),
    }, vs


def build_constraints(cfg, geom):
    """Build PDE, Dirichlet, zero-flux and insulated-wall constraints on v.

    Args:
        cfg (DictConfig): Runtime config.
        geom (Dict): Geometry dict from `build_geometry`.

    Returns:
        Dict of constraint name to `ppsci.constraint.Constraint`.
    """
    equation = {"laplace_v": LaplaceV(dim=2)}
    terms = _u_singularity_terms(cfg)
    loss = ppsci.loss.MSELoss("sum")
    dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    constraints = {}

    # PDE residual of the smooth part v on the copper interior
    constraints["EQ"] = ppsci.constraint.InteriorConstraint(
        equation["laplace_v"].equations,
        {"laplace_v": 0},
        geom["interior"],
        {**dataloader_cfg, "batch_size": cfg.NPOINT_INTERIOR},
        loss,
        name="EQ",
    )

    # Dirichlet v = -u_s on the VRM pad ring
    constraints["VRM"] = ppsci.constraint.BoundaryConstraint(
        {"v": lambda out: out["v"]},
        {
            "v": lambda input_dict: -_u_s(input_dict["x"], input_dict["y"], terms)[
                0
            ].reshape(-1, 1)
        },
        geom["vrm"],
        {**dataloader_cfg, "batch_size": cfg.NPOINT_VRM},
        loss,
        name="VRM",
    )

    # Zero-flux on each load pad ring: the singularity term u_s carries the
    # prescribed injection flux, so the smooth part has zero normal flux.
    def dv_dn(out):
        return (
            jacobian(out["v"], out["x"]) * out["normal_x"]
            + jacobian(out["v"], out["y"]) * out["normal_y"]
        )

    for name, _, _, r, amp in cfg.PADS:
        constraints[f"PAD_{name}"] = ppsci.constraint.BoundaryConstraint(
            {f"flux_{name}": dv_dn},
            {f"flux_{name}": 0},
            geom["pads"][name],
            {**dataloader_cfg, "batch_size": cfg.NPOINT_PER_PAD},
            loss,
            name=f"PAD_{name}",
        )

    # Insulated walls of the plane: dv/dn = -grad(u_s) . n with n the normal
    # sampled by ppsci on the board edge, the slot wall and the holes.
    def insulated_label(input_dict):
        _, dus_dx, dus_dy = _u_s(input_dict["x"], input_dict["y"], terms)
        nx = input_dict["normal_x"].astype("float64")
        ny = input_dict["normal_y"].astype("float64")
        return -(dus_dx * nx + dus_dy * ny)

    # up-weight the insulated walls: without it the optimizer settles in a
    # v ~ 0 local minimum that satisfies the cheap VRM/PAD terms but leaves
    # the far-field amplitude of v (up to ~3 near the right-hand loads)
    # completely unlearned
    wall_weight = cfg.TRAIN.get("wall_weight", 1.0)
    for tag, g, npoint in (
        ("EDGE", geom["board_edge"], cfg.NPOINT_EDGE),
        ("SLOT", geom["slot_wall"], cfg.NPOINT_SLOT),
        ("HOLE", geom["holes_union"], cfg.NPOINT_HOLE),
    ):
        if g is None:
            continue
        key = f"{tag.lower()}_flux"
        constraints[tag] = ppsci.constraint.BoundaryConstraint(
            {key: dv_dn},
            {key: insulated_label},
            g,
            {**dataloader_cfg, "batch_size": npoint},
            loss,
            weight_dict={key: wall_weight} if wall_weight != 1.0 else None,
            name=tag,
        )

    # sparse interior anchors from the FDM reference: pins the far-field
    # amplitude of v that the pure boundary constraints cannot convey
    n_anchor = cfg.TRAIN.get("anchor_points", 0)
    anchor = build_anchor_points(cfg, n_anchor)
    if anchor is not None:
        anchor_input, anchor_label = anchor
        # anchor_weight is folded into sqrt-scaled labels/outputs because
        # SupervisedConstraint has no weight_dict parameter: MSE of
        # (sqrt(w)*pred, sqrt(w)*label) equals w * MSE(pred, label)
        w_sqrt = float(np.sqrt(cfg.TRAIN.get("anchor_weight", 1.0)))
        constraints["ANCHOR"] = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": anchor_input,
                    "label": {"v_anchor": w_sqrt * anchor_label},
                },
                "batch_size": len(anchor_label),
                "sampler": {
                    "name": "BatchSampler",
                    "drop_last": False,
                    "shuffle": True,
                },
                "num_workers": 0,
            },
            ppsci.loss.MSELoss("sum"),
            {"v_anchor": lambda out: w_sqrt * out["v"]},
            name="ANCHOR",
        )
    return constraints


def build_validator(cfg, geom):
    """Build a supervised validator against the FDM reference solution.

    Args:
        cfg (DictConfig): Runtime config.
        geom (Dict): Geometry dict from `build_geometry`.

    Returns:
        Tuple (validator dict, FDM result dict).
    """
    fd = solve_ir_drop_fd(
        (cfg.BOARD[0], cfg.BOARD[1]),
        cfg.SLOT,
        cfg.VRM,
        cfg.PADS,
        cfg.HOLES_COORD,
        cfg.HOLE_R,
        cfg.FDM_N,
    )
    logger.message(
        f"FDM reference: grid h=1/{cfg.FDM_N}, "
        f"VRM absorbed current = {fd['vrm_in']:.6f} (conservation, must be 1)"
    )
    X, Y, U, copper = fd["X"], fd["Y"], fd["U"], fd["copper"]
    # exclude nodes inside the VRM and load pad disks: the log singularity
    # expansion u_s diverges at the pad centers and does not represent u
    # inside the disks (grid nodes can coincide with a pad center, giving
    # u_s = inf and poisoning every metric)
    inside_disk = np.zeros_like(copper)
    for _, cx, cy, r_pad, _ in cfg.PADS:
        inside_disk |= (X - cx) ** 2 + (Y - cy) ** 2 <= r_pad**2
    inside_disk |= (X - cfg.VRM[1]) ** 2 + (Y - cfg.VRM[2]) ** 2 <= cfg.VRM[3] ** 2
    valid = copper & ~inside_disk
    input_dict = {
        "x": X[valid].reshape([-1, 1]).astype("float32"),
        "y": Y[valid].reshape([-1, 1]).astype("float32"),
    }
    label_dict = {"u_ref": U[valid].reshape([-1, 1]).astype("float32")}
    terms = _u_singularity_terms(cfg)
    validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict,
                "label": label_dict,
            },
            "batch_size": cfg.EVAL.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": False},
        },
        ppsci.loss.MSELoss("mean"),
        # reconstruct u = v + u_s at every FDM copper node
        {"u_ref": lambda out: out["v"] + _u_s_paddle(out, terms)[0]},
        metric={"L2Rel": ppsci.metric.L2Rel(), "MaxAE": ppsci.metric.MaxAE()},
        name="FDM_ref",
    )
    return {"FDM_ref": validator}, fd


def report_pad_voltages(solver, cfg, fd):
    """Predict pad ring voltages and log the effective resistance table.

    Args:
        solver (ppsci.solver.Solver): Trained solver.
        cfg (DictConfig): Runtime config.
        fd (Dict): FDM reference result (for comparison columns).
    """
    i_total = sum(p[4] for p in cfg.PADS)
    u0 = i_total / cfg.SIGMA_S
    terms = _u_singularity_terms(cfg)
    fd_pads = fd_eval_pads(fd, cfg.VRM, cfg.PADS)
    logger.message(
        f"{'pad':6s} {'I [A]':>6s} {'PINN V [mV]':>12s} {'FDM V [mV]':>11s} "
        f"{'PINN R [mOhm]':>14s} {'FDM R [mOhm]':>13s}"
    )
    for name, cx, cy, r, amp in cfg.PADS:
        theta = np.linspace(0.0, 2.0 * np.pi, 65, endpoint=False)
        rr = r + 1.5 / cfg.FDM_N
        pts = {
            "x": (cx + rr * np.cos(theta)).reshape([-1, 1]).astype("float32"),
            "y": (cy + rr * np.sin(theta)).reshape([-1, 1]).astype("float32"),
        }
        pred = solver.predict(pts, None)
        # u = v + u_s evaluated at the ring points
        pred_np = {k: np.asarray(v) for k, v in pred.items()}
        us, _, _ = _u_s(pts["x"], pts["y"], terms)
        u_pred = pred_np["v"] + us.astype("float32").reshape(-1, 1)
        v_pinn = float(u_pred.mean()) * u0
        v_fd = fd_pads[name][0] * u0
        logger.message(
            f"{name:6s} {amp:6.2f} {1e3 * v_pinn:12.4f} {1e3 * v_fd:11.4f} "
            f"{1e3 * v_pinn / amp:14.3f} {1e3 * v_fd / amp:13.3f}"
        )


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry and constraints
    geom = build_geometry(cfg)
    constraint = build_constraints(cfg, geom)

    # set optimizer (optional exponential lr decay when TRAIN.lr_scheduler set)
    if cfg.TRAIN.get("lr_scheduler", None):
        lr = ppsci.optimizer.lr_scheduler.ExponentialDecay(**cfg.TRAIN.lr_scheduler)()
        optimizer = ppsci.optimizer.Adam(lr)(model)
    else:
        optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator (FDM reference)
    validator, fd = build_validator(cfg, geom)

    # set visualizer (u = v + u_s reconstructed on the fly)
    vis_terms = _u_singularity_terms(cfg)
    vis_input = geom["interior"].sample_interior(cfg.NPOINT_VIS, random="Halton")
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerVtu(
            vis_input,
            {"u": lambda d: d["v"] + _u_s_paddle(d, vis_terms)[0]},
            batch_size=cfg.EVAL.batch_size,
            num_timestamps=1,
            prefix="result_u",
        )
    }

    # initialize solver; optional MTL aggregator for conflicting gradients
    aggregator_name = cfg.TRAIN.get("loss_aggregator", "Sum")
    n_cst = len(constraint)
    aggregator = {
        "Sum": lambda: None,
        "GradNorm": lambda: ppsci.loss.mtl.GradNorm(model, num_losses=n_cst),
        "PCGrad": lambda: ppsci.loss.mtl.PCGrad(model),
        "NTK": lambda: ppsci.loss.mtl.NTK(model, num_losses=n_cst),
        "Relobralo": lambda: ppsci.loss.mtl.Relobralo(num_losses=n_cst),
    }[aggregator_name]()

    # stage 1: anchor-only warm start. Joint training of the anchor
    # supervision and the boundary/PDE terms settles into a tug-of-war
    # plateau (L2Rel ~ 0.4); fitting the sparse anchors first lands the
    # network inside the basin of the true solution, after which stage 2
    # fine-tunes with the full physics without leaving it.
    n_anchor = cfg.TRAIN.get("anchor_points", 0)
    warm_epochs = cfg.TRAIN.get("warm_start_epochs", 0)
    if n_anchor > 0 and warm_epochs > 0:
        anchor_solver = ppsci.solver.Solver(
            model,
            {"ANCHOR": constraint["ANCHOR"]},
            cfg.output_dir,
            ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model),
            epochs=warm_epochs,
            iters_per_epoch=1,
            geom=geom,
        )
        anchor_solver.train()
        logger.message(f"stage 1 (anchor warm start, {warm_epochs} epochs) finished")

    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        loss_aggregator=aggregator,
    )
    # train model
    solver.train()
    # evaluate and report the pad voltage table
    solver.eval()
    report_pad_voltages(solver, cfg, fd)


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry and validator (FDM reference)
    geom = build_geometry(cfg)
    validator, fd = build_validator(cfg, geom)

    # set visualizer (u = v + u_s reconstructed on the fly)
    vis_terms = _u_singularity_terms(cfg)
    vis_input = geom["interior"].sample_interior(cfg.NPOINT_VIS, random="Halton")
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerVtu(
            vis_input,
            {"u": lambda d: d["v"] + _u_s_paddle(d, vis_terms)[0]},
            batch_size=cfg.EVAL.batch_size,
            num_timestamps=1,
            prefix="result_u",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()
    solver.visualize()
    report_pad_voltages(solver, cfg, fd)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    geom = build_geometry(cfg)
    input_dict = geom["interior"].sample_interior(cfg.NPOINT_VIS, random="Halton")
    output_dict = predictor.predict(
        {key: input_dict[key] for key in cfg.MODEL.input_keys}, cfg.INFER.batch_size
    )

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    # save result
    ppsci.visualize.save_vtu_from_dict(
        osp.join(cfg.output_dir, "ems_ir_drop_pred.vtu"),
        {**input_dict, **output_dict},
        input_dict.keys(),
        cfg.MODEL.output_keys,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="ems_ir_drop.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
