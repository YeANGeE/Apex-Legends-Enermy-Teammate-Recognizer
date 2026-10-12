"""Reproduce WebUI model timing repeatedly on one image without saving outputs."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from c1_webui import DEFAULT_THRESHOLD, DetectionEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    engine = DetectionEngine()
    timings: list[float] = []
    print(f"Image: {args.image}")
    print(f"GPU: {engine.gpu_name}")
    for index in range(args.runs):
        result = engine.detect(
            image_bytes,
            DEFAULT_THRESHOLD,
            args.image.name,
            save_output=False,
        )
        timing = float(result["inference_ms"])
        timings.append(timing)
        print(f"run {index + 1:02d}: {timing:6.1f} ms")

    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    print(f"median: {statistics.median(timings):.1f} ms")
    print(f"p95:    {ordered[p95_index]:.1f} ms")
    print(f"min:    {min(timings):.1f} ms")
    print(f"max:    {max(timings):.1f} ms")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
