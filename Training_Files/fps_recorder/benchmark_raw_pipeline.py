"""Measure the recorder's raw BGR pipe, CPU scale, and NVENC path.

This reuses one decoded frame, so it isolates throughput rather than visual
quality or live capture. A live game test is still required for acceptance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import AppConfig
from .utils import resolve_executable
from .video_encoder import build_ffmpeg_command


def decoded_frames(source: Path, count: int = 8) -> list[np.ndarray]:
    probe = subprocess.run(
        [resolve_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(source)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    result = subprocess.run(
        [resolve_executable("ffmpeg"), "-v", "error", "-i", str(source),
         "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        check=True, capture_output=True,
    )
    return list(np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3))


def run_case(
    config: AppConfig,
    images: list[np.ndarray],
    frames: int,
    output: Path,
    *,
    video_filter: str | None = None,
    realtime_fps: int | None = None,
) -> dict:
    height, width = images[0].shape[:2]
    command = build_ffmpeg_command(
        config, output, input_width=width, input_height=height
    )
    if video_filter is not None:
        command[command.index("-vf") + 1] = video_filter
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdin is not None
    payloads = [memoryview(np.ascontiguousarray(image)).cast("B") for image in images]
    started = time.perf_counter()
    try:
        for index in range(frames):
            if realtime_fps is not None:
                delay = started + index / realtime_fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            process.stdin.write(payloads[index % len(payloads)])
        fed = time.perf_counter()
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        result = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    ended = time.perf_counter()
    probe = subprocess.run(
        [resolve_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=avg_frame_rate,nb_read_packets",
         "-of", "json", str(output)],
        capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0] if probe.returncode == 0 else {}
    return {
        "input": f"{width}x{height} bgr24",
        "filter": command[command.index("-vf") + 1],
        "realtime_fps": realtime_fps,
        "frames": frames,
        "raw_GB": round(len(payloads[0]) * frames / 1e9, 3),
        "feed_fps": round(frames / (fed - started), 2),
        "total_fps": round(frames / (ended - started), 2),
        "feed_seconds": round(fed - started, 3),
        "flush_seconds": round(ended - fed, 3),
        "return_code": result,
        "output_frame_rate": stream.get("avg_frame_rate"),
        "output_packets": int(stream.get("nb_read_packets", -1)),
        "output_bytes": output.stat().st_size if output.exists() else 0,
        "error": stderr.decode(errors="replace")[-2000:] if result else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument(
        "--realtime-seconds", type=float,
        help="encode only the 4K path paced at output FPS for this duration",
    )
    parser.add_argument(
        "--gpu-compare", action="store_true",
        help="also benchmark CUDA scaling after CPU BGR-to-NV12 conversion",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.realtime_seconds is not None and args.realtime_seconds <= 0:
        parser.error("--realtime-seconds must be positive")
    config = AppConfig.load(args.config)
    images = decoded_frames(args.source)
    if images[0].shape[:2] != (1080, 1920):
        parser.error("source must be 1920x1080")
    with tempfile.TemporaryDirectory(prefix="fps_raw_bench_") as directory:
        root = Path(directory)
        images_4k = [np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1) for frame in images]
        if args.realtime_seconds is not None:
            frames = round(args.realtime_seconds * config.encoder.output_fps)
            results = [run_case(
                config, images_4k, frames, root / "4k_realtime.mkv",
                realtime_fps=config.encoder.output_fps,
            )]
        else:
            results = [run_case(config, images, args.frames, root / "1080p.mkv")]
            results.append(run_case(config, images_4k, args.frames, root / "4k.mkv"))
        if args.gpu_compare and args.realtime_seconds is None:
            results.append(run_case(
                config, images_4k, args.frames, root / "4k_cuda.mkv",
                video_filter="format=nv12,hwupload_cuda,scale_cuda=1920:1080:format=nv12",
            ))
    print(json.dumps(results, indent=2))
    return 0 if all(row["return_code"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
