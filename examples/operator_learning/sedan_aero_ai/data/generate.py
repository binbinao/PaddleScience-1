"""
Synthetic flow field data generator for sedan external aerodynamics.

Generates simplified 3D flow fields around a car-like geometry using
analytical and semi-empirical models. This is for testing the pipeline;
real training should use CFD-generated data (OpenFOAM/SU2/Star-CCM+).
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class SedanGeometry:
    """Simplified sedan geometry using parametric representation."""

    def __init__(
        self,
        length: float = 4.5,
        width: float = 1.8,
        height: float = 1.4,
        wheelbase: float = 2.7,
        hood_ratio: float = 0.25,      # hood length / total length
        cabin_ratio: float = 0.35,      # cabin length / total length
        trunk_ratio: float = 0.25,      # trunk length / total length
        windshield_angle: float = 30.0, # degrees from horizontal
        rear_window_angle: float = 25.0,
        ground_clearance: float = 0.15,
    ):
        self.length = length
        self.width = width
        self.height = height
        self.wheelbase = wheelbase
        self.hood_ratio = hood_ratio
        self.cabin_ratio = cabin_ratio
        self.trunk_ratio = trunk_ratio
        self.windshield_angle = np.radians(windshield_angle)
        self.rear_window_angle = np.radians(rear_window_angle)
        self.ground_clearance = ground_clearance

    def get_sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute Signed Distance Function for the sedan geometry.

        Args:
            x, y, z: Coordinate arrays (can be broadcast-compatible).

        Returns:
            SDF values: negative inside, positive outside.
        """
        half_l = self.length / 2
        half_w = self.width / 2

        # Base body (rectangular prism with rounded edges)
        # Car body defined from -half_l to +half_l in x
        body_sdf = self._body_sdf(x, y, z)

        # Cabin/roof (raised section)
        cabin_sdf = self._cabin_sdf(x, y, z)

        # Wheels (subtract from body)
        wheel_sdf = self._wheel_sdf(x, y, z)

        # Ground plane
        ground_sdf = z - self.ground_clearance

        # Combine: union of body and cabin, minus wheels, above ground
        sdf = np.minimum(np.minimum(body_sdf, cabin_sdf), ground_sdf)
        sdf = np.maximum(sdf, -wheel_sdf)  # subtract wheels

        return sdf

    def _body_sdf(self, x, y, z):
        """Main body as a rounded box."""
        half_l = self.length / 2
        half_w = self.width / 2
        hood_height = 0.75  # hood height
        trunk_height = 0.7  # trunk height

        # Body top profile: hood -> windshield -> roof -> rear window -> trunk
        top_profile = self._top_profile(x)

        qx = np.abs(x) - half_l
        qy = np.abs(y) - half_w
        qz = z - top_profile

        # Rounded box SDF
        exterior = np.maximum(np.maximum(qx, qy), qz)
        interior = np.minimum(np.maximum(qx, qy), 0.0) + np.minimum(qz, 0.0)
        return exterior + interior

    def _top_profile(self, x):
        """Height profile of the car top surface along x-axis."""
        half_l = self.length / 2
        hood_height = 0.75
        roof_height = 1.35
        trunk_height = 0.7

        # Normalize x to [-1, 1] along car length
        x_norm = x / half_l

        # Piecewise profile
        result = np.where(
            x_norm < -1 + self.hood_ratio,
            hood_height,  # front bumper to hood
            np.where(
                x_norm < -1 + self.hood_ratio + 0.1,
                hood_height + (roof_height - hood_height) * (x_norm + 1 - self.hood_ratio) / 0.1,
                np.where(  # hood to windshield
                    x_norm < -1 + self.hood_ratio + 0.1 + 0.05,
                    roof_height,
                    np.where(  # windshield
                        x_norm < 1 - self.trunk_ratio - 0.1,
                        roof_height,  # roof
                        np.where(
                            x_norm < 1 - self.trunk_ratio,
                            roof_height + (trunk_height - roof_height) * (x_norm - (1 - self.trunk_ratio - 0.1)) / 0.1,
                            trunk_height,  # rear window to trunk
                        )
                    )
                )
            )
        )
        return result

    def _cabin_sdf(self, x, y, z):
        """Cabin/roof raised section."""
        half_l = self.length / 2
        half_w = self.width / 2 * 0.85  # cabin narrower than body

        cabin_start = -half_l + self.hood_ratio * self.length + 0.1 * self.length
        cabin_end = half_l - self.trunk_ratio * self.length - 0.1 * self.length
        roof_height = 1.35

        # Cabin bounding box
        qx = np.maximum(cabin_start - x, x - cabin_end)
        qy = np.abs(y) - half_w
        qz = z - roof_height

        return np.maximum(np.maximum(qx, qy), qz)

    def _wheel_sdf(self, x, y, z):
        """Wheel wells (subtracted from body)."""
        half_l = self.length / 2
        wheel_radius = 0.32
        wheel_width = 0.22

        # Front wheel position (center)
        front_wheel_x = -half_l + 0.3 * self.length
        # Rear wheel position (center)
        rear_wheel_x = half_l - 0.3 * self.length

        wheel_y = self.width / 2 - 0.05  # slightly inside body width
        wheel_z = wheel_radius + self.ground_clearance

        # Front-left wheel
        fl = np.sqrt(
            ((x - front_wheel_x) / wheel_radius)**2 +
            ((y + wheel_y) / wheel_width)**2 +
            ((z - wheel_z) / wheel_radius)**2
        ) - 1.0

        # Front-right wheel
        fr = np.sqrt(
            ((x - front_wheel_x) / wheel_radius)**2 +
            ((y - wheel_y) / wheel_width)**2 +
            ((z - wheel_z) / wheel_radius)**2
        ) - 1.0

        # Rear-left wheel
        rl = np.sqrt(
            ((x - rear_wheel_x) / wheel_radius)**2 +
            ((y + wheel_y) / wheel_width)**2 +
            ((z - wheel_z) / wheel_radius)**2
        ) - 1.0

        # Rear-right wheel
        rr = np.sqrt(
            ((x - rear_wheel_x) / wheel_radius)**2 +
            ((y - wheel_y) / wheel_width)**2 +
            ((z - wheel_z) / wheel_radius)**2
        ) - 1.0

        return np.minimum(np.minimum(fl, fr), np.minimum(rl, rr))


class FlowFieldGenerator:
    """Generate synthetic 3D flow fields around a sedan."""

    def __init__(self, config: dict):
        self.cfg = config
        self.Re = config["PHYSICS"]["Re"]
        self.U_inf = config["PHYSICS"]["U_inf"]
        self.nu = config["PHYSICS"]["nu"]
        self.rho = config["PHYSICS"]["rho"]

        self.bbox = config["DOMAIN"]["bbox"]
        self.nx = config["DOMAIN"]["nx"]
        self.ny = config["DOMAIN"]["ny"]
        self.nz = config["DOMAIN"]["nz"]

        self.geometry = SedanGeometry(
            length=config["PHYSICS"]["L_ref"],
            width=1.8,
            height=1.4,
        )

    def generate_sample(self, u_inf: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate a single flow field sample.

        Args:
            u_inf: Inlet velocity. Uses config default if None.

        Returns:
            dict with keys: 'sdf', 'u', 'v', 'w', 'p', 'coords', 'u_inf'
        """
        if u_inf is None:
            u_inf = self.U_inf

        # Generate grid
        x = np.linspace(self.bbox[0], self.bbox[1], self.nx)
        y = np.linspace(self.bbox[2], self.bbox[3], self.ny)
        z = np.linspace(self.bbox[4], self.bbox[5], self.nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Compute SDF
        sdf = self.geometry.get_sdf(X, Y, Z)

        # Generate synthetic flow field
        u, v, w, p = self._synthetic_flow(X, Y, Z, sdf, u_inf)

        return {
            "sdf": sdf.astype(np.float32),
            "u": u.astype(np.float32),
            "v": v.astype(np.float32),
            "w": w.astype(np.float32),
            "p": p.astype(np.float32),
            "coords": np.stack([X, Y, Z], axis=-1).astype(np.float32),
            "u_inf": u_inf,
        }

    def _synthetic_flow(self, X, Y, Z, sdf, u_inf):
        """
        Generate synthetic flow field using simplified potential flow + wake model.

        This is NOT physically accurate — it provides a plausible flow structure
        for testing the model pipeline. Real training MUST use CFD data.
        """
        # Freestream
        u_fs = u_inf * np.ones_like(X)

        # Body effect: slow down near car surface
        inside_car = sdf < 0.05
        boundary_layer = (sdf > 0.05) & (sdf < 0.5)
        u_body = np.where(
            inside_car, 0.0,
            np.where(boundary_layer, u_inf * (1 - np.exp(-sdf / 0.05)), u_inf)
        )
        u = u_fs * 0.3 + u_body * 0.7

        # Wake: velocity deficit behind car
        car_center_x = 0.0
        wake_start = 2.5  # start of wake behind car
        wake_mask = X > wake_start
        wake_strength = 0.3 * u_inf * np.exp(-((X - wake_start) / 5.0)**2)
        wake_decay_y = np.exp(-(Y**2) / (2 * 1.0**2))
        wake_decay_z = np.exp(-((Z - 0.7)**2) / (2 * 0.7**2))
        u = u - wake_strength * wake_decay_y * wake_decay_z * wake_mask

        # Lateral velocity (flow around car)
        v = u_inf * 0.1 * np.sign(Y) * np.exp(-np.abs(Y) / 2.0) * \
            np.exp(-((X - car_center_x)**2) / 9.0)

        # Vertical velocity (upwash over hood, downwash over trunk)
        hood_region = (X > -2.0) & (X < -1.0) & (np.abs(Y) < 1.0)
        trunk_region = (X > 1.0) & (X < 2.5) & (np.abs(Y) < 1.0)
        w = np.zeros_like(Z)
        w[hood_region] = u_inf * 0.05 * (Z[hood_region] / 1.5)
        w[trunk_region] = -u_inf * 0.08 * (Z[trunk_region] / 1.5)

        # Pressure (simplified Bernoulli + wake depression)
        p_dynamic = 0.5 * 1.225 * u_inf**2
        p = -p_dynamic * 0.1 * (1 - (u / u_inf)**2)
        p = p - p.min()  # normalize to positive

        # Set interior to zero
        u[inside_car] = 0.0
        v[inside_car] = 0.0
        w[inside_car] = 0.0
        p[inside_car] = 0.0

        return u, v, w, p

    def generate_dataset(self, n_samples: int, u_inf_range: Tuple[float, float] = (20.0, 40.0)) -> list:
        """Generate multiple samples with varying inlet velocities."""
        samples = []
        for i in range(n_samples):
            u_inf = np.random.uniform(*u_inf_range)
            sample = self.generate_sample(u_inf=u_inf)
            samples.append(sample)
        return samples


def create_fno_dataset(samples: list) -> Dict[str, np.ndarray]:
    """
    Convert flow field samples to FNO training format.

    FNO expects:
        Input:  (N, C_in, D, H, W)  — C_in=1 (SDF as input condition)
        Output: (N, C_out, D, H, W) — C_out=4 (u, v, w, p)

    where N=batch, D=depth(z), H=height(y), W=width(x).
    """
    n = len(samples)
    input_data = np.stack([s["sdf"] for s in samples])  # (N, D, H, W)
    input_data = input_data[:, np.newaxis, ...]           # (N, 1, D, H, W)

    u = np.stack([s["u"] for s in samples])
    v = np.stack([s["v"] for s in samples])
    w = np.stack([s["w"] for s in samples])
    p = np.stack([s["p"] for s in samples])

    output_data = np.stack([u, v, w, p], axis=1)  # (N, 4, D, H, W)

    return {
        "input": input_data.astype(np.float32),
        "output": output_data.astype(np.float32),
    }


def save_dataset(data: Dict[str, np.ndarray], save_dir: str, prefix: str):
    """Save dataset to disk."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / f"{prefix}.npz"
    np.savez_compressed(filepath, **data)
    print(f"Saved {prefix} dataset to {filepath}: input {data['input'].shape}, output {data['output'].shape}")
    return filepath
