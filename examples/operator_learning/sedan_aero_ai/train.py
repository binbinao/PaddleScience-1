#!/usr/bin/env python3
"""
Training script for sedan external aerodynamics AI surrogate model.

This script trains a Fourier Neural Operator (FNO) or CNN-based surrogate
to predict 3D flow fields (velocity + pressure) around a sedan geometry,
replacing traditional CFD solvers with 100-1000x faster AI inference.

Usage:
    python train.py                          # Train with default config
    python train.py --model fno              # Use FNO architecture
    python train.py --model cnn              # Use 3D CNN architecture
    python train.py --epochs 1000 --lr 5e-4  # Custom training params
    python train.py --eval                  # Evaluate only
    python train.py --infer                 # Inference only

Reference:
    - Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
    - PaddleScience: https://github.com/PaddlePaddle/PaddleScience
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import paddle
import paddle.nn as nn
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import SedanAeroFNO, SedanAeroCNN, build_model, count_parameters
from data.generate import FlowFieldGenerator, create_fno_dataset, save_dataset
from utils.helpers import (
    AverageMeter,
    MetricsTracker,
    setup_training_env,
    CheckpointManager,
    format_time,
    memory_usage_mb,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI surrogate for sedan aerodynamics")

    parser.add_argument("--config", type=str, default="configs/sedan_fno.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default="fno", choices=["fno", "cnn"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluation mode")
    parser.add_argument("--infer", action="store_true",
                        help="Inference mode")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or gpu")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
) -> paddle.io.DataLoader:
    """Create a PaddlePaddle DataLoader from numpy data."""
    dataset = paddle.io.TensorDataset([
        paddle.to_tensor(data["input"]),
        paddle.to_tensor(data["output"]),
    ])

    loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    return loader


def train_epoch(
    model: nn.Layer,
    loader: paddle.io.DataLoader,
    optimizer: paddle.optimizer.Optimizer,
    loss_fn: nn.Layer,
    epoch: int,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.cpu() if device == "cpu" else inputs
        targets = targets.cpu() if device == "cpu" else targets

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.shape[0])

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}: loss = {loss.item():.6f}")

    return {"train_loss": losses.avg}


def validate(model, loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    losses = AverageMeter()

    with paddle.no_grad():
        for inputs, targets in loader:
            inputs = inputs.cpu() if device == "cpu" else inputs
            targets = targets.cpu() if device == "cpu" else targets

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.shape[0])

    return {"val_loss": losses.avg}


def train(config: dict, args):
    """Main training loop."""
    device = args.device

    # Override config with CLI args
    if args.epochs:
        config["TRAIN"]["epochs"] = args.epochs
    if args.batch_size:
        config["TRAIN"]["batch_size"] = args.batch_size
    if args.lr:
        config["TRAIN"]["lr_scheduler"]["learning_rate"] = args.lr

    config["MODEL"]["arch"] = args.model

    # Setup environment
    env = setup_training_env({**config, "output_dir": args.output_dir, "seed": args.seed})

    # Generate or load data
    print("=" * 60)
    print("Preparing dataset...")
    print("=" * 60)

    data_dir = Path(args.output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    generator = FlowFieldGenerator(config)

    use_synthetic = config["DATA"].get("use_synthetic", True)
    if use_synthetic:
        print("Generating synthetic flow field data...")
        train_samples = config["DATA"]["train_samples"]
        val_samples = config["DATA"]["val_samples"]

        train_data_raw = generator.generate_dataset(train_samples, u_inf_range=(25.0, 35.0))
        val_data_raw = generator.generate_dataset(val_samples, u_inf_range=(28.0, 32.0))

        train_data = create_fno_dataset(train_data_raw)
        val_data = create_fno_dataset(val_data_raw)

        save_dataset(train_data, str(data_dir), "train")
        save_dataset(val_data, str(data_dir), "val")
    else:
        print(f"Loading data from {config['DATA']['data_dir']}")
        train_data = dict(np.load(Path(config["DATA"]["data_dir"]) / "train.npz"))
        val_data = dict(np.load(Path(config["DATA"]["data_dir"]) / "val.npz"))

    print(f"  Train: input {train_data['input'].shape}, output {train_data['output'].shape}")
    print(f"  Val:   input {val_data['input'].shape}, output {val_data['output'].shape}")

    # Create dataloaders
    batch_size = config["TRAIN"]["batch_size"]
    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)

    # Build model
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)

    model = build_model(config)
    n_params = count_parameters(model)
    print(f"  Architecture: {config['MODEL'].get('arch', 'fno')}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")

    # Loss and optimizer
    loss_fn = nn.MSELoss()

    lr_cfg = config["TRAIN"]["lr_scheduler"]
    lr = lr_cfg["learning_rate"]
    if args.lr:
        lr = args.lr

    # Learning rate scheduler: step decay
    lr_schedule = paddle.optimizer.lr.StepDecay(
        learning_rate=lr,
        step_size=lr_cfg.get("step_size", 100),
        gamma=lr_cfg.get("gamma", 0.5),
    )

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr_schedule,
        weight_decay=config["TRAIN"].get("weight_decay", 1e-4),
    )

    # Checkpoint manager
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_manager = CheckpointManager(str(ckpt_dir), max_keep=5)

    # Load pretrained if specified
    start_epoch = 0
    if args.pretrained:
        print(f"\nLoading pretrained model: {args.pretrained}")
        model.set_state_dict(paddle.load(args.pretrained))

    # Metrics tracker
    metrics_tracker = MetricsTracker()

    # Training loop
    epochs = config["TRAIN"]["epochs"]
    eval_freq = config["TRAIN"].get("eval_freq", 50)
    save_freq = config["TRAIN"].get("save_freq", 100)

    print("\n" + "=" * 60)
    print(f"Starting training ({epochs} epochs)")
    print("=" * 60)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, epoch, device
        )
        metrics_tracker.update(train_metrics, epoch)

        # Evaluate
        if (epoch + 1) % eval_freq == 0 or epoch == epochs - 1:
            val_metrics = validate(model, val_loader, loss_fn, device)
            metrics_tracker.update(val_metrics, epoch)

            val_loss = val_metrics["val_loss"]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            print(f"\nEpoch {epoch+1}/{epochs}: "
                  f"train_loss={train_metrics['train_loss']:.6f}, "
                  f"val_loss={val_loss:.6f} "
                  f"{'[BEST]' if is_best else ''} "
                  f"time={format_time(time.time() - epoch_start)}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            ckpt_manager.save(
                model, optimizer, epoch,
                {"train_loss": train_metrics["train_loss"]},
                is_best=(epoch == epochs - 1 or val_loss == best_val_loss),
            )

        # Learning rate step
        lr_schedule.step()

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete! Total time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)

    # Save final model
    final_path = Path(args.output_dir) / "final_model.pdparams"
    paddle.save(model.state_dict(), str(final_path))
    print(f"Final model saved to {final_path}")

    # Save metrics
    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_tracker.save(str(metrics_path))

    return model


def evaluate(config: dict, args):
    """Evaluation mode."""
    print("=" * 60)
    print("Evaluation Mode")
    print("=" * 60)

    config["MODEL"]["arch"] = args.model

    # Load data
    data_path = Path(args.output_dir) / "data" / "val.npz"
    if data_path.exists():
        val_data = dict(np.load(str(data_path)))
        print(f"Loaded validation data: {val_data['input'].shape}")
    else:
        print("No validation data found. Generating...")
        generator = FlowFieldGenerator(config)
        samples = generator.generate_dataset(config["DATA"]["val_samples"])
        val_data = create_fno_dataset(samples)

    # Build model
    model = build_model(config)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Load weights
    if args.pretrained:
        model.set_state_dict(paddle.load(args.pretrained))
    else:
        pretrained_path = Path(args.output_dir) / "final_model.pdparams"
        if pretrained_path.exists():
            model.set_state_dict(paddle.load(str(pretrained_path)))
        else:
            print("WARNING: No pretrained model found. Using random weights.")

    # Evaluate
    model.eval()
    loss_fn = nn.MSELoss()

    loader = create_dataloader(val_data, batch_size=1, shuffle=False)

    total_loss = 0.0
    with paddle.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item()

    avg_loss = total_loss / len(loader)
    print(f"\nValidation MSE Loss: {avg_loss:.6f}")
    print(f"Relative Error: {np.sqrt(avg_loss):.6f}")

    return {"val_loss": avg_loss}


def inference(config: dict, args):
    """Inference mode — predict flow field and benchmark speed."""
    print("=" * 60)
    print("Inference & Benchmarking Mode")
    print("=" * 60)

    config["MODEL"]["arch"] = args.model

    # Build model
    model = build_model(config)

    # Load weights
    if args.pretrained:
        model.set_state_dict(paddle.load(args.pretrained))
    else:
        pretrained_path = Path(args.output_dir) / "final_model.pdparams"
        if pretrained_path.exists():
            model.set_state_dict(paddle.load(str(pretrained_path)))
        else:
            print("WARNING: No pretrained model found. Using random weights.")

    model.eval()

    # Generate a test sample
    print("\nGenerating test geometry...")
    generator = FlowFieldGenerator(config)
    sample = generator.generate_sample(u_inf=config["PHYSICS"]["U_inf"])

    # Prepare input
    input_data = sample["sdf"][np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
    input_tensor = paddle.to_tensor(input_data, dtype="float32")

    # AI Inference timing
    print("\nRunning AI inference...")
    n_warmup = 3
    n_timing = 10

    # Warmup
    with paddle.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

    # Timed inference
    paddle.device.cuda.synchronize() if paddle.is_compiled_with_cuda() else None
    start = time.perf_counter()
    with paddle.no_grad():
        for _ in range(n_timing):
            _ = model(input_tensor)
    paddle.device.cuda.synchronize() if paddle.is_compiled_with_cuda() else None
    elapsed = time.perf_counter() - start

    ai_time = elapsed / n_timing

    # Get prediction
    with paddle.no_grad():
        prediction = model(input_tensor)

    pred_np = prediction.numpy()[0]  # (4, D, H, W)
    u_pred = pred_np[0]
    v_pred = pred_np[1]
    w_pred = pred_np[2]
    p_pred = pred_np[3]

    # Benchmark
    print("\n" + "=" * 60)
    print("AI Inference Results")
    print("=" * 60)
    print(f"  Inference time: {ai_time * 1000:.2f} ms")
    print(f"  Throughput:     {1.0 / ai_time:.1f} samples/sec")
    print(f"  Grid size:      {input_data.shape[-3]}x{input_data.shape[-2]}x{input_data.shape[-1]}")

    # Compare with typical CFD time
    # A typical RANS simulation for external aero: ~10-60 min on 32 cores
    typical_cfd_time = 30 * 60  # 30 minutes
    speedup = typical_cfd_time / ai_time
    print(f"\n  Typical CFD (RANS, 32 cores): ~{typical_cfd_time / 60:.0f} min")
    print(f"  AI Inference:                 {ai_time:.4f} s")
    print(f"  Speedup:                      {speedup:.0f}x")

    import json
    benchmark_results = {
        "ai_inference_time_ms": ai_time * 1000,
        "throughput_per_sec": 1.0 / ai_time,
        "grid_size": list(input_data.shape[-3:]),
        "typical_cfd_time_s": typical_cfd_time,
        "speedup_ratio": speedup,
        "model_params": count_parameters(model),
    }

    bench_path = Path(args.output_dir) / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(benchmark_results, f, indent=2, default=float)
    print(f"\nBenchmark saved to {bench_path}")

    # Save prediction
    pred_path = Path(args.output_dir) / "prediction.npz"
    np.savez_compressed(
        str(pred_path),
        u=u_pred,
        v=v_pred,
        w=w_pred,
        p=p_pred,
        coords=sample["coords"],
        sdf=sample["sdf"],
    )
    print(f"Prediction saved to {pred_path}")

    return benchmark_results


def main():
    args = parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config not found at {config_path}, using defaults")
        config = {
            "MODEL": {"arch": args.model},
            "PHYSICS": {"U_inf": 30.0, "Re": 2e6, "rho": 1.225, "nu": 1.5e-5, "L_ref": 4.5, "A_ref": 2.2},
            "DOMAIN": {"bbox": [-15, 25, -8, 8, 0, 8], "nx": 64, "ny": 32, "nz": 16},
            "TRAIN": {"epochs": args.epochs or 100, "batch_size": args.batch_size or 2,
                       "lr_scheduler": {"learning_rate": args.lr or 1e-3, "step_size": 50, "gamma": 0.5}},
            "DATA": {"use_synthetic": True, "train_samples": 200, "val_samples": 50},
        }
    else:
        config = load_config(str(config_path))

    # Set device
    if args.device == "gpu" and paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    print("=" * 60)
    print("Sedan External Aerodynamics — AI Surrogate Model")
    print(f"Framework: PaddlePaddle {paddle.__version__}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model.upper()}")
    print("=" * 60)

    if args.eval:
        results = evaluate(config, args)
    elif args.infer:
        results = inference(config, args)
    else:
        model = train(config, args)
        # Run quick inference after training
        config["MODEL"]["arch"] = args.model
        results = inference(config, args)

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
