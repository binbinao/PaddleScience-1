"""
Utility functions for sedan aerodynamics AI surrogate model.
"""

import os
import time
import json
import numpy as np
import paddle
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict


class AverageMeter:
    """Track running average of values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track training/validation metrics."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, metrics: Dict[str, float], step: int):
        for k, v in metrics.items():
            self.history[k].append((step, v))

    def get_latest(self, key: str) -> float:
        if key in self.history and self.history[key]:
            return self.history[key][-1][1]
        return float("nan")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            self.history = json.load(f)


def compute_relative_l2_error(pred: paddle.Tensor, target: paddle.Tensor) -> float:
    """Compute relative L2 error."""
    diff = pred - target
    return float(
        (paddle.norm(diff) / (paddle.norm(target) + 1e-8)).item()
    )


def compute_drag_coefficient(
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    sdf: np.ndarray,
    u_inf: float,
    rho: float,
    A_ref: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[float, float]:
    """
    Estimate drag and lift coefficients from flow field.

    Simplified integration of pressure and shear stress on car surface.
    For accurate Cd/Cl, use proper surface integration on the car mesh.

    Returns:
        (Cd, Cl) tuple
    """
    # Find car surface (SDF near zero)
    surface_mask = (np.abs(sdf) < 0.05) & (sdf > -0.01)

    if not surface_mask.any():
        return 0.0, 0.0

    # Dynamic pressure
    q = 0.5 * rho * u_inf**2

    # Pressure drag (simplified)
    p_force = np.sum(p[surface_mask]) * dx * dy  # projection on y-z plane

    Cd = p_force / (q * A_ref) if q * A_ref > 0 else 0.0
    Cl = 0.0  # Simplified — needs proper surface integration

    return float(Cd), float(Cl)


def compute_speedup(
    cfd_time_seconds: float,
    ai_time_seconds: float,
    cfd_cores: int = 1,
    ai_device: str = "GPU",
) -> Dict[str, float]:
    """
    Compute speedup metrics.

    Args:
        cfd_time_seconds: Time for traditional CFD simulation.
        ai_time_seconds: Time for AI inference.
        cfd_cores: Number of CPU cores used for CFD.
        ai_device: Device used for AI inference.

    Returns:
        dict with speedup ratios.
    """
    speedup = cfd_time_seconds / ai_time_seconds if ai_time_seconds > 0 else float("inf")

    return {
        "cfd_time_s": cfd_time_seconds,
        "ai_time_s": ai_time_seconds,
        "speedup_ratio": speedup,
        "cfd_cores": cfd_cores,
        "ai_device": ai_device,
    }


def setup_training_env(config: dict) -> dict:
    """Set up training environment."""
    # Set random seeds
    seed = config.get("seed", 42)
    paddle.seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(config.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    return {"output_dir": str(output_dir), "seed": seed}


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, save_dir: str, max_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []

    def save(self, model, optimizer, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint."""
        path = self.save_dir / f"checkpoint_epoch_{epoch}.pdparams"
        opt_path = self.save_dir / f"checkpoint_epoch_{epoch}.pdopt"

        paddle.save(model.state_dict(), str(path))
        paddle.save(optimizer.state_dict(), str(opt_path))

        self.checkpoints.append((epoch, str(path)))

        # Clean old checkpoints
        while len(self.checkpoints) > self.max_keep:
            old_epoch, old_path = self.checkpoints.pop(0)
            old_file = Path(old_path)
            if old_file.exists():
                old_file.unlink()

        # Save best
        if is_best:
            best_path = self.save_dir / "best_model.pdparams"
            paddle.save(model.state_dict(), str(best_path))

    def load(self, model, optimizer=None, path: Optional[str] = None) -> int:
        """Load checkpoint. Returns epoch number."""
        if path is None:
            # Load latest
            best_path = self.save_dir / "best_model.pdparams"
            if best_path.exists():
                path = str(best_path)
            elif self.checkpoints:
                path = self.checkpoints[-1][1]
            else:
                return 0

        if path and os.path.exists(path):
            model.set_state_dict(paddle.load(path))
            return 0
        return 0


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def memory_usage_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        if paddle.is_compiled_with_cuda():
            return paddle.device.cuda.memory_allocated() / 1024**2
    except:
        pass
    return 0.0
