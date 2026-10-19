"""Run a one-batch end-to-end smoke test with ``python -m snn_slam``."""

import argparse

import torch

from .models import AllSNN_SLAM
from .training import synthetic_batch, train_step


def main() -> None:
    parser = argparse.ArgumentParser(description="All-SNN SLAM smoke training")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = AllSNN_SLAM().to(args.device)
    batch = synthetic_batch(2, args.steps, args.height, args.width, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    metrics = train_step(model, batch, optimizer)
    print({key: round(float(value), 6) for key, value in metrics.as_dict().items()})


if __name__ == "__main__":
    main()

