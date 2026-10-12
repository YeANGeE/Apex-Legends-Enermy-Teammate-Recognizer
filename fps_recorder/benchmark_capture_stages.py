"""Profile DXcam's 4K capture stages at 60 and 120 FPS without encoding.

This desktop-only benchmark does not measure in-game frame pacing. Run it when
no other recorder owns the DXGI duplication output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import psutil

from .clock import QpcClock
from .config import AppConfig
from .dxcam_capture import DXcamCapture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", choices=("60", "120", "both"), default="both")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    from dxcam.core.stagesurf import StageSurface
    from dxcam.dxcam import DXCamera
    from dxcam.processor import Processor

    config = AppConfig.load(args.config)
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    maxima: dict[str, float] = {}
    originals = []

    def instrument(cls, method_name: str, label: str) -> None:
        original = getattr(cls, method_name)

        def wrapped(self, *method_args, **method_kwargs):
            start = time.perf_counter()
            try:
                return original(self, *method_args, **method_kwargs)
            finally:
                elapsed = time.perf_counter() - start
                counts[label] = counts.get(label, 0) + 1
                totals[label] = totals.get(label, 0.0) + elapsed
                maxima[label] = max(maxima.get(label, 0.0), elapsed)

        setattr(cls, method_name, wrapped)
        originals.append((cls, method_name, original))

    instrument(DXCamera, "_copy_region_to_stage", "gpu_copy_submit")
    instrument(StageSurface, "map", "staging_map")
    instrument(Processor, "process_into", "bgr_convert")
    instrument(StageSurface, "unmap", "staging_unmap")

    results = []
    try:
        for fps in ((60, 120) if args.fps == "both" else (int(args.fps),)):
            counts.clear()
            totals.clear()
            maxima.clear()
            selected = 0

            def on_frame(_frame):
                nonlocal selected
                selected += 1

            capture = DXcamCapture(
                replace(config.capture, fps=fps), QpcClock(), on_frame,
                output_fps=60,
            )
            process = psutil.Process()
            cpu_start = process.cpu_times()
            started = time.perf_counter()
            try:
                capture.start()
                time.sleep(args.seconds)
            finally:
                capture.stop()
            wall = time.perf_counter() - started
            cpu_end = process.cpu_times()
            stats = capture.snapshot_stats()
            results.append({
                "capture_fps_target": fps,
                "wall_seconds": round(wall, 3),
                "captured_frames": stats.frames,
                "captured_fps": round(stats.frames / wall, 2),
                "selected_frames": selected,
                "ring_overwrites": stats.dxcam_ring_overwrites,
                "capture_errors": stats.errors,
                "process_cpu_core_percent": round(
                    100 * (
                        cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
                    ) / wall, 2,
                ),
                "stages": {
                    label: {
                        "calls": counts.get(label, 0),
                        "mean_ms": round(
                            1000 * totals.get(label, 0.0) / counts[label], 3
                        ) if counts.get(label, 0) else 0,
                        "max_ms": round(1000 * maxima.get(label, 0.0), 3),
                        "total_seconds": round(totals.get(label, 0.0), 3),
                    }
                    for label in (
                        "gpu_copy_submit", "staging_map", "bgr_convert",
                        "staging_unmap",
                    )
                },
            })
    finally:
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)

    print(json.dumps(results, indent=2))
    return 0 if all(row["capture_errors"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
