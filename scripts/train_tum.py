"""Train and validate the all-SNN model on a TUM RGB-D sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

from snn_slam import AllSNN_SLAM, TUMRGBDWindowDataset, collate_tum_batches
from snn_slam.training import (
    evaluate_epoch,
    evaluate_slam_epoch,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all-SNN SLAM on TUM RGB-D")
    parser.add_argument("--root", required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--fx-px", type=float, default=517.3)
    parser.add_argument("--fy-px", type=float, default=516.5)
    parser.add_argument("--sensor-width", type=int, default=640)
    parser.add_argument("--sensor-height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="tum_snn_slam.pt")
    parser.add_argument("--resume", action="store_true", help="resume model and optimizer from --checkpoint")
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = TUMRGBDWindowDataset(
        args.root,
        window=args.window,
        stride=args.stride,
        image_size=(args.height, args.width),
        max_frames=args.max_frames,
    )
    if len(dataset) < 2:
        raise ValueError("need at least two windows for train/validation split")
    split = max(1, int(len(dataset) * 0.8))
    split = min(split, len(dataset) - 1)
    # Leave one window-length gap where possible so overlapping windows do not
    # put nearly identical frames in both train and validation sets.
    gap = max(1, (args.window + args.stride - 1) // args.stride - 1)
    validation_start = split + gap
    if validation_start >= len(dataset):
        validation_start = split
    train_data = Subset(dataset, range(split))
    validation_data = Subset(dataset, range(validation_start, len(dataset)))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "collate_fn": collate_tum_batches,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)

    model = AllSNN_SLAM(
        feature_channels=args.feature_channels,
        map_size=args.map_size,
        focal_length_normalized=(
            2.0 * args.fx_px / args.sensor_width,
            2.0 * args.fy_px / args.sensor_height,
        ),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, map_location=args.device)
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"resumed checkpoint={args.checkpoint} epoch={start_epoch}")
    max_batches = args.max_batches if args.max_batches > 0 else None
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, max_batches=max_batches)
        validation_metrics = evaluate_epoch(model, validation_loader, args.device, max_batches=max_batches)
        trajectory_metrics = evaluate_slam_epoch(model, validation_loader, args.device, max_batches=max_batches)
        train_text = ", ".join(f"{key}={value:.5f}" for key, value in train_metrics.items())
        validation_text = ", ".join(f"{key}={value:.5f}" for key, value in validation_metrics.items())
        trajectory_text = ", ".join(f"{key}={value:.5f}" for key, value in trajectory_metrics.items())
        print(f"epoch={epoch + 1} train: {train_text}")
        print(f"epoch={epoch + 1} validation: {validation_text}")
        print(f"epoch={epoch + 1} trajectory: {trajectory_text}")
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            epoch=epoch + 1,
            extra={
                "config": vars(args),
                "trajectory_metrics": trajectory_metrics,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            },
        )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        save_checkpoint(args.checkpoint, model, optimizer, epoch=start_epoch, extra={"config": vars(args)})
    print(f"saved checkpoint: {checkpoint.resolve()}")


if __name__ == "__main__":
    main()

