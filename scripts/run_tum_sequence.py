"""Run the SNN-SLAM pipeline on a TUM RGB-D image sequence.

The script uses the dataset adapter's documented pseudo-event bridge. It is a
development/evaluation tool, not a replacement for a real event-camera
stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from snn_slam import AllSNN_SLAM, TUMRGBDWindowDataset, evaluate_output
from snn_slam.training import TrainBatch, train_step


def _batch(sample: TrainBatch) -> TrainBatch:
    return TrainBatch(
        events=sample.events.unsqueeze(0),
        poses=sample.poses.unsqueeze(0) if sample.poses is not None else None,
        depth=sample.depth.unsqueeze(0) if sample.depth is not None else None,
        depth_mask=sample.depth_mask.unsqueeze(0) if sample.depth_mask is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-SNN SLAM on a TUM RGB-D sequence")
    parser.add_argument("--root", required=True, help="TUM sequence directory or its extracted parent")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--eval-windows", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=0, help="gradient steps on initial windows")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="", help="optional output checkpoint path")
    args = parser.parse_args()

    dataset = TUMRGBDWindowDataset(
        args.root,
        window=args.window,
        image_size=(args.height, args.width),
        max_frames=args.max_frames,
    )
    model = AllSNN_SLAM(feature_channels=32, map_size=32).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.train_steps):
        sample = _batch(dataset[step % len(dataset)])
        sample = TrainBatch(
            events=sample.events.to(args.device),
            poses=sample.poses.to(args.device) if sample.poses is not None else None,
            depth=sample.depth.to(args.device) if sample.depth is not None else None,
            depth_mask=sample.depth_mask.to(args.device) if sample.depth_mask is not None else None,
        )
        metrics = train_step(model, sample, optimizer)
        print(f"train step={step + 1} total={float(metrics.total):.5f} pose={float(metrics.pose):.5f} depth={float(metrics.depth):.5f}")

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(args)}, checkpoint)

    model.eval()
    with torch.no_grad():
        for index in range(min(args.eval_windows, len(dataset))):
            sample = _batch(dataset[index])
            output = model(sample.events.to(args.device), reset=True)
            metrics = evaluate_output(
                output,
                sample.poses.to(args.device),
                sample.depth.to(args.device) if sample.depth is not None else None,
                sample.depth_mask.to(args.device) if sample.depth_mask is not None else None,
            )
            formatted = {key: round(float(value), 6) for key, value in metrics.items()}
            print(f"eval window={index}: {formatted}")


if __name__ == "__main__":
    main()

