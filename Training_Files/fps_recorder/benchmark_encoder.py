from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .monitor import read_encoder_utilization
from .utils import resolve_executable


@dataclass(frozen=True)
class Setting:
    name: str
    target_mbps: float
    max_mbps: float
    cq: int


@dataclass
class Result:
    setting: str
    average_mbps: float
    storage_MBps: float
    size_per_minute_MB: float
    size_for_3_minutes_MB: float
    encoder_fps: float
    elapsed_seconds: float
    drops: int
    average_encoder_utilization: float | None
    output_file: str
    error: str = ""


DEFAULT_SETTINGS = [
    Setting("vbr_6M_cq32", 6, 7.5, 32),
    Setting("vbr_8M_cq31", 8, 9.5, 31),
    Setting("vbr_10M_cq30", 10, 12, 30),
    Setting("vbr_12M_cq29", 12, 14, 29),
    Setting("vbr_14M_cq28", 14, 15.5, 28),
    Setting("cq26_cap14M", 0, 14, 26),
    Setting("cq30_cap12M", 0, 12, 30),
    Setting("cq34_cap10M", 0, 10, 34),
]


def probe_video(path: Path) -> tuple[float, int, float]:
    command = [
        resolve_executable("ffprobe"),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))
    stream = data["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    duration = float(data["format"]["duration"])
    frames = int(stream.get("nb_read_frames") or round(duration * fps))
    return duration, frames, fps


def encode_one(source: Path, output: Path, setting: Setting) -> Result:
    source_duration, source_frames, _ = probe_video(source)
    command = [
        resolve_executable("ffmpeg"),
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p5",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-b:v",
        f"{setting.target_mbps:g}M",
        "-maxrate",
        f"{setting.max_mbps:g}M",
        "-bufsize",
        f"{setting.max_mbps * 2:g}M",
        "-cq",
        str(setting.cq),
        "-multipass",
        "qres",
        "-rc-lookahead",
        "20",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-aq-strength",
        "8",
        "-g",
        "240",
        "-bf",
        "3",
        "-fps_mode",
        "passthrough",
        str(output),
    ]
    stop_sample = threading.Event()
    gpu_samples: list[float] = []

    def sample_gpu() -> None:
        while not stop_sample.wait(0.25):
            value = read_encoder_utilization()
            if value is not None:
                gpu_samples.append(value)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    stop_sample.set()
    sampler.join(timeout=1.0)
    error = "" if process.returncode == 0 else process.stderr[-4000:]
    size = output.stat().st_size if output.exists() else 0
    average_mbps = size * 8 / source_duration / 1_000_000
    drop_matches = re.findall(r"drop=\s*(\d+)", process.stderr)
    drops = int(drop_matches[-1]) if drop_matches else 0
    return Result(
        setting=setting.name,
        average_mbps=average_mbps,
        storage_MBps=average_mbps / 8,
        size_per_minute_MB=average_mbps / 8 * 60,
        size_for_3_minutes_MB=average_mbps / 8 * 180,
        encoder_fps=source_frames / elapsed,
        elapsed_seconds=elapsed,
        drops=drops,
        average_encoder_utilization=(
            sum(gpu_samples) / len(gpu_samples) if gpu_samples else None
        ),
        output_file=str(output),
        error=error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark AV1 NVENC settings on one representative sample"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        encode_one(args.source, args.output_dir / f"{item.name}.mkv", item)
        for item in DEFAULT_SETTINGS
    ]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, indent=2)
    print("| Setting | Mbps | MB/s | MB/min | 3 min MB | Enc FPS | Drops | NVENC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        util = "n/a" if result.average_encoder_utilization is None else f"{result.average_encoder_utilization:.1f}%"
        print(
            f"| {result.setting} | {result.average_mbps:.2f} | "
            f"{result.storage_MBps:.2f} | {result.size_per_minute_MB:.1f} | "
            f"{result.size_for_3_minutes_MB:.1f} | {result.encoder_fps:.1f} | "
            f"{result.drops} | {util} |"
        )
    return 0 if all(not result.error for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

