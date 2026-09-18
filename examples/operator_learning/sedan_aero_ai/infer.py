#!/usr/bin/env python3
"""
Inference and visualization for sedan external aerodynamics AI surrogate.

After training, use this script to:
  1. Predict flow fields for new conditions
  2. Benchmark inference speed vs traditional CFD
  3. Visualize results (velocity, pressure, streamlines)
  4. Compute aerodynamic coefficients (Cd, Cl)

Usage:
    python infer.py --model-path outputs/final_model.pdparams
    python infer.py --u-inf 25.0 --u-inf 35.0  # sweep inlet velocities
    python infer.py --benchmark                    # speed benchmark
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import paddle
from pathlib import Path
from typing import Dict, Optional, List

sys.path.insert(0, str(Path(__file__).parent))

from models import SedanAeroFNO, build_model, count_parameters
from data.generate import FlowFieldGenerator
from utils.helpers import format_time


def parse_args():
    parser = argparse.ArgumentParser(description="AI surrogate inference for sedan aerodynamics")

    parser.add_argument("--model-path", type=str, default="outputs/final_model.pdparams",
                        help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/sedan_fno.yaml",
                        help="Config file")
    parser.add_argument("--u-inf", type=float, nargs="+", default=[30.0],
                        help="Inlet velocity(ies) in m/s (30 = ~108 km/h)")
    parser.add_argument("--model", type=str, default="fno", choices=["fno", "cnn"],
                        help="Model architecture")
    parser.add_argument("--output-dir", type=str, default="./inference_results",
                        help="Output directory for results")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run detailed speed benchmark")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization data")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or gpu")

    return parser.parse_args()


def load_model(model_path: str, config: dict, device: str) -> paddle.nn.Layer:
    """Load trained model."""
    config["MODEL"]["arch"] = config.get("MODEL", {}).get("arch", "fno")
    model = build_model(config)

    if os.path.exists(model_path):
        model.set_state_dict(paddle.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}, using random weights")

    model.eval()
    return model


def predict_flow(
    model: paddle.nn.Layer,
    sdf: np.ndarray,
    u_inf: float,
) -> Dict[str, np.ndarray]:
    """
    Predict flow field for a given geometry and inlet velocity.

    Args:
        model: Trained surrogate model.
        sdf: Signed distance function of geometry (D, H, W).
        u_inf: Inlet velocity in m/s.

    Returns:
        dict with u, v, w, p arrays.
    """
    # Normalize input
    input_data = sdf[np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
    input_tensor = paddle.to_tensor(input_data, dtype="float32")

    with paddle.no_grad():
        output = model(input_tensor)

    output_np = output.numpy()[0]  # (4, D, H, W)

    return {
        "u": output_np[0] * u_inf / 30.0,  # Scale by inlet velocity
        "v": output_np[1] * u_inf / 30.0,
        "w": output_np[2] * u_inf / 30.0,
        "p": output_np[3],
    }


def benchmark_inference(
    model: paddle.nn.Layer,
    input_shape: tuple,
    n_warmup: int = 5,
    n_timing: int = 50,
) -> Dict[str, float]:
    """
    Detailed inference speed benchmark.

    Args:
        model: Trained model.
        input_shape: Tuple (D, H, W) for input.
        n_warmup: Number of warmup iterations.
        n_timing: Number of timing iterations.

    Returns:
        dict with timing statistics.
    """
    print("\n" + "=" * 60)
    print("Speed Benchmark")
    print("=" * 60)

    # Create random input with correct shape
    dummy_input = paddle.randn([1, 1] + list(input_shape))

    # Warmup
    print(f"Warming up ({n_warmup} iterations)...")
    with paddle.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Synchronize if GPU
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()

    # Timing
    print(f"Benchmarking ({n_timing} iterations)...")
    times = []
    with paddle.no_grad():
        for i in range(n_timing):
            input_tensor = paddle.randn([1, 1] + list(input_shape))
            start = time.perf_counter()
            _ = model(input_tensor)
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # CFD comparison
    # Typical RANS: 10-60 min on 32 cores for ~5M cells
    # Typical LES: 24-72 hours on 128+ cores
    cfd_rans_time = 30 * 60     # 30 min
    cfd_les_time = 48 * 3600    # 48 hours

    results = {
        "input_shape": list(input_shape),
        "n_params": count_parameters(model),
        "mean_time_ms": mean_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "throughput_per_sec": 1.0 / mean_time,
        "throughput_per_min": 60.0 / mean_time,
        "speedup_vs_rans_30min": cfd_rans_time / mean_time,
        "speedup_vs_les_48h": cfd_les_time / mean_time,
    }

    print(f"\n  Input shape:     {list(input_shape)}")
    print(f"  Model params:    {results['n_params']:,}")
    print(f"  Mean time:       {mean_time*1000:.2f} ms")
    print(f"  Std:             {std_time*1000:.2f} ms")
    print(f"  Throughput:      {results['throughput_per_sec']:.1f} samples/sec")
    print(f"\n  Speedup vs RANS (30 min, 32 cores):  {results['speedup_vs_rans_30min']:.0f}x")
    print(f"  Speedup vs LES (48 hours, 128 cores): {results['speedup_vs_les_48h']:.0f}x")

    return results


def sweep_velocity(
    model: paddle.nn.Layer,
    generator: FlowFieldGenerator,
    velocities: List[float],
    output_dir: str,
):
    """Sweep across inlet velocities and predict flow fields."""
    print("\n" + "=" * 60)
    print(f"Velocity Sweep: {velocities} m/s")
    print("=" * 60)

    # Generate base geometry
    sample = generator.generate_sample(u_inf=30.0)
    sdf = sample["sdf"]

    results = []
    for u_inf in velocities:
        print(f"\n  U_inf = {u_inf:.1f} m/s ({u_inf * 3.6:.0f} km/h)")
        flow = predict_flow(model, sdf, u_inf)

        # Save per-velocity results
        save_path = Path(output_dir) / f"flow_uinf_{u_inf:.0f}.npz"
        np.savez_compressed(str(save_path), **flow, sdf=sdf, u_inf=u_inf)

        # Quick stats
        results.append({
            "u_inf": u_inf,
            "max_velocity": float(np.max(np.sqrt(flow["u"]**2 + flow["v"]**2 + flow["w"]**2))),
            "max_pressure": float(np.max(flow["p"])),
            "avg_wake_velocity": float(np.mean(flow["u"][-20:, ...])),
        })
        print(f"    Max velocity: {results[-1]['max_velocity']:.1f} m/s")
        print(f"    Max pressure: {results[-1]['max_pressure']:.1f} Pa")
        print(f"    Wake velocity: {results[-1]['avg_wake_velocity']:.1f} m/s")

    # Save sweep summary
    summary_path = Path(output_dir) / "velocity_sweep.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSweep summary saved to {summary_path}")

    return results


def compute_aero_coefficients(
    flow: Dict[str, np.ndarray],
    sdf: np.ndarray,
    u_inf: float,
    rho: float = 1.225,
    A_ref: float = 2.2,
) -> Dict[str, float]:
    """
    Estimate aerodynamic coefficients from predicted flow field.

    Simplified computation using pressure integration on the car surface.
    For production use, implement proper surface mesh integration.
    """
    u = flow["u"]
    v = flow["v"]
    w = flow["w"]
    p = flow["p"]

    # Dynamic pressure
    q = 0.5 * rho * u_inf**2

    # Car surface (where SDF is near 0)
    surface = (np.abs(sdf) < 0.1) & (sdf > -0.05)

    if surface.any():
        # Approximate surface normal using SDF gradient
        # (simplified: assume faces are roughly x-oriented for drag)
        p_surface = p[surface]
        drag_force = np.mean(p_surface) * np.sum(surface)  # simplified
        Cd = drag_force / (q * A_ref) if q * A_ref > 0 else 0.0
    else:
        Cd = 0.0

    return {
        "Cd": float(Cd),
        "Cl": 0.0,  # Placeholder
        "u_inf": u_inf,
        "q_dynamic": float(q),
        "Reynolds": float(u_inf * 4.5 / 1.5e-5),
    }


def main():
    args = parse_args()

    # Set device
    if args.device == "gpu" and paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    print("=" * 60)
    print("Sedan Aerodynamics — AI Surrogate Inference")
    print(f"Device: {args.device}")
    print(f"Model:  {args.model.upper()}")
    print("=" * 60)

    # Load config
    import yaml
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "MODEL": {"arch": args.model, "in_channels": 1, "out_channels": 4,
                       "n_modes_depth": 12, "n_modes_height": 12, "n_modes_width": 12,
                       "hidden_channels": 64, "n_layers": 4, "domain_padding": 0.0},
            "PHYSICS": {"U_inf": 30.0, "Re": 2e6, "rho": 1.225, "nu": 1.5e-5,
                         "L_ref": 4.5, "A_ref": 2.2},
            "DOMAIN": {"bbox": [-15, 25, -8, 8, 0, 8], "nx": 64, "ny": 32, "nz": 16},
        }

    # Load model
    model = load_model(args.model_path, config, args.device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate geometry
    generator = FlowFieldGenerator(config)

    # Benchmark if requested
    if args.benchmark:
        input_shape = (config["DOMAIN"]["nz"], config["DOMAIN"]["ny"], config["DOMAIN"]["nx"])
        bench_results = benchmark_inference(model, input_shape)
        bench_path = output_dir / "benchmark.json"
        with open(bench_path, "w") as f:
            json.dump(bench_results, f, indent=2, default=float)
        print(f"\nBenchmark saved to {bench_path}")

    # Velocity sweep
    if len(args.u_inf) > 1:
        sweep_velocity(model, generator, args.u_inf, str(output_dir))
    else:
        u_inf = args.u_inf[0]
        print(f"\nPredicting flow at U_inf = {u_inf:.1f} m/s ({u_inf * 3.6:.0f} km/h)")

        sample = generator.generate_sample(u_inf=u_inf)
        sdf = sample["sdf"]

        # Predict
        start = time.perf_counter()
        flow = predict_flow(model, sdf, u_inf)
        elapsed = time.perf_counter() - start
        print(f"Inference time: {elapsed * 1000:.2f} ms")

        # Compute aero coefficients
        aero = compute_aero_coefficients(
            flow, sdf, u_inf,
            rho=config["PHYSICS"]["rho"],
            A_ref=config["PHYSICS"]["A_ref"],
        )
        print(f"\nEstimated Aerodynamic Coefficients:")
        print(f"  Cd = {aero['Cd']:.4f}")
        print(f"  Cl = {aero['Cl']:.4f}")
        print(f"  Re = {aero['Reynolds']:.1e}")
        print(f"  q  = {aero['q_dynamic']:.1f} Pa")

        # Save results
        result_path = output_dir / "prediction.npz"
        np.savez_compressed(
            str(result_path),
            u=flow["u"], v=flow["v"], w=flow["w"], p=flow["p"],
            sdf=sdf, coords=sample["coords"], u_inf=u_inf,
        )
        print(f"\nResults saved to {result_path}")

        aero_path = output_dir / "aerodynamics.json"
        with open(aero_path, "w") as f:
            json.dump(aero, f, indent=2, default=float)
        print(f"Aero coefficients saved to {aero_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
