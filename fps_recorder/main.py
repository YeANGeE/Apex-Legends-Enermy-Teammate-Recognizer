from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import signal
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import psutil

from . import __version__
from .clock import QpcClock
from .config import AppConfig
from .dxcam_capture import DXcamCapture
from .episode_manager import EpisodeManager
from .gameinput_capture import GameInputCapture
from .xinput_capture import XInputCapture
from .monitor import read_encoder_utilization, read_gpu_name


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous DXGI + GameInput RB-triggered FPS recorder"
    )
    parser.add_argument("--config", default="config.yaml", help="YAML configuration")
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Optional bounded run for diagnostics; normal use is unbounded",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--capture-fps", type=int, choices=(60, 120), default=None,
        help="Temporary capture-rate override for 60-vs-120 FPS comparison",
    )
    parser.add_argument(
        "--stop-file", type=Path, default=None,
        help="Stop cleanly when this initially absent file is created",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop_file = args.stop_file.resolve() if args.stop_file is not None else None
    if stop_file is not None and stop_file.exists():
        raise FileExistsError(f"Stop file already exists: {stop_file}")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = AppConfig.load(args.config)
    if args.capture_fps is not None:
        config = replace(config, capture=replace(config.capture, fps=args.capture_fps))
        config.validate()
    clock = QpcClock()
    stop_requested = threading.Event()
    capture: DXcamCapture | None = None
    gameinput: GameInputCapture | None = None
    xinput: XInputCapture | None = None
    gpu_name = read_gpu_name()

    def metadata_provider() -> dict:
        calibration = gameinput.calibration if gameinput else None
        return {
            "recorder_version": __version__,
            "platform": platform.platform(),
            "python_version": sys.version,
            "process_id": os.getpid(),
            "qpc_frequency": clock.frequency,
            "dxgi_timestamp_source": "DXGI_OUTDUPL_FRAME_INFO.LastPresentTime",
            "gameinput_timestamp_source": "IGameInputReading.GetTimestamp",
            "gameinput_calibration": (
                calibration.to_dict() if calibration is not None else None
            ),
            "capture_region": list(capture.active_region) if capture else None,
            "gpu": gpu_name,
            "capture_stats": (
                asdict(capture.snapshot_stats()) if capture is not None else None
            ),
            "gameinput_stats": (
                asdict(gameinput.snapshot_stats()) if gameinput is not None else None
            ),
            "xinput_stats": asdict(xinput.snapshot_stats()) if xinput else None,
        }

    manager = EpisodeManager(config, clock, metadata_provider)
    capture = DXcamCapture(
        config.capture, clock, manager.on_frame, output_fps=config.encoder.output_fps
    )
    gameinput = GameInputCapture(config.gameinput, clock, manager.on_gamepad)
    if config.xinput.enabled:
        xinput = XInputCapture(config.xinput, manager.on_xinput,
                              manager.on_input_disconnect)

    def request_stop(_signum=None, _frame=None) -> None:
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    started = time.monotonic()
    last_status = 0.0
    last_gpu_sample = 0.0
    encoder_util: float | None = None
    exit_code = 0
    metrics_path = Path(config.episode.output_dir) / f"session_{time.strftime('%Y%m%d_%H%M%S')}_metrics.csv"
    metrics_handle = None
    try:
        capture.start()
        gameinput.start()
        if xinput:
            xinput.start()
        LOGGER.info(
            "Recorder active: region=%s, source=%dx%d, capture=%d FPS, stored=1920x1080@%d, output=%s",
            capture.active_region,
            capture.source_width,
            capture.source_height,
            config.capture.fps,
            config.encoder.output_fps,
            Path(config.episode.output_dir),
        )
        if gameinput.calibration:
            LOGGER.info("GameInput/QPC calibration: %s", gameinput.calibration)
        metrics_handle = metrics_path.open("w", newline="", encoding="utf-8")
        fields = [
            "elapsed_seconds", "capture_frames", "capture_fps", "capture_fps_interval",
            "ring_overwrites", "source_cadence_gaps", "source_cadence_gaps_interval",
            "duplicate_timestamps", "duplicate_timestamps_interval", "capture_errors",
            "gameinput_readings", "gameinput_rate_hz", "gameinput_next_calls",
            "gameinput_empty_next_calls", "gameinput_max_gap_ms", "gameinput_errors",
            "gameinput_device_rebinds",
            "xinput_readings", "xinput_state_changes", "xinput_poll_hz",
            "xinput_change_hz", "xinput_max_gap_ms", "xinput_errors",
            "xinput_disconnects",
            "episode_state", "episodes_started", "episodes_completed", "episodes_failed",
            "encoder_queue_depth", "encoder_queue_overflows", "pending_encoders",
            "pending_encoder_queue_depth",
            "storage_rate_MBps", "encoder_utilization_percent", "process_tree_rss_MB",
            "process_tree_cpu_percent",
            "process_tree_cpu_percent_of_host",
        ]
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
        metrics_writer.writeheader()
        process = psutil.Process()
        process.cpu_percent()
        seen_processes = {process.pid: process}
        previous_capture_sample = (started, 0, 0, 0)
        LOGGER.info("Session metrics: %s", metrics_path)
        while not stop_requested.wait(0.05):
            if stop_file is not None and stop_file.exists():
                LOGGER.info("Stop file detected: %s", stop_file)
                stop_requested.set()
                break
            now = time.monotonic()
            if args.run_seconds is not None and now - started >= args.run_seconds:
                stop_requested.set()
                break
            if now - last_gpu_sample >= config.status.gpu_sample_seconds:
                encoder_util = read_encoder_utilization()
                last_gpu_sample = now
            if now - last_status >= config.status.interval_seconds:
                c = capture.snapshot_stats()
                g = gameinput.snapshot_stats()
                x = xinput.snapshot_stats() if xinput else None
                e = manager.snapshot_stats()
                previous_time, previous_frames, previous_gaps, previous_duplicates = (
                    previous_capture_sample
                )
                sample_seconds = now - previous_time
                capture_fps_interval = (
                    (c.frames - previous_frames) / sample_seconds
                    if sample_seconds > 0 else 0.0
                )
                cadence_gaps_interval = c.source_cadence_gaps - previous_gaps
                duplicates_interval = c.duplicate_timestamps - previous_duplicates
                previous_capture_sample = (
                    now, c.frames, c.source_cadence_gaps, c.duplicate_timestamps
                )
                children = process.children(recursive=True)
                for child in children:
                    if child.pid not in seen_processes:
                        child.cpu_percent()
                        seen_processes[child.pid] = child
                processes = [process, *(seen_processes[child.pid] for child in children)]
                rss_bytes = 0
                cpu_percent = 0.0
                for member in processes:
                    try:
                        rss_bytes += member.memory_info().rss
                        cpu_percent += member.cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                metrics_writer.writerow({
                    "elapsed_seconds": round(now - started, 3),
                    "capture_frames": c.frames,
                    "capture_fps": c.actual_fps,
                    "capture_fps_interval": round(capture_fps_interval, 2),
                    "ring_overwrites": c.dxcam_ring_overwrites,
                    "source_cadence_gaps": c.source_cadence_gaps,
                    "source_cadence_gaps_interval": cadence_gaps_interval,
                    "duplicate_timestamps": c.duplicate_timestamps,
                    "duplicate_timestamps_interval": duplicates_interval,
                    "capture_errors": c.errors,
                    "gameinput_readings": g.readings,
                    "gameinput_rate_hz": g.rate_hz,
                    "gameinput_next_calls": g.next_reading_calls,
                    "gameinput_empty_next_calls": g.empty_next_reading_calls,
                    "gameinput_max_gap_ms": g.max_reading_gap_ms,
                    "gameinput_errors": g.errors,
                    "gameinput_device_rebinds": g.device_rebinds,
                    "xinput_readings": x.readings if x else None,
                    "xinput_state_changes": x.state_changes if x else None,
                    "xinput_poll_hz": x.rate_hz if x else None,
                    "xinput_change_hz": x.change_rate_hz if x else None,
                    "xinput_max_gap_ms": x.max_poll_gap_ms if x else None,
                    "xinput_errors": x.errors if x else None,
                    "xinput_disconnects": x.disconnects if x else None,
                    "episode_state": e.state,
                    "episodes_started": e.episodes_started,
                    "episodes_completed": e.episodes_completed,
                    "episodes_failed": e.episodes_failed,
                    "encoder_queue_depth": e.active_encoder_queue,
                    "encoder_queue_overflows": e.active_encoder_overflows,
                    "pending_encoders": e.pending_encoders,
                    "pending_encoder_queue_depth": e.pending_encoder_queue,
                    "storage_rate_MBps": e.active_storage_rate_MBps,
                    "encoder_utilization_percent": encoder_util,
                    "process_tree_rss_MB": round(rss_bytes / 1_000_000, 2),
                    "process_tree_cpu_percent": round(cpu_percent, 2),
                    "process_tree_cpu_percent_of_host": round(
                        cpu_percent / (psutil.cpu_count() or 1), 2
                    ),
                })
                metrics_handle.flush()
                gpu_text = "n/a" if encoder_util is None else f"{encoder_util:.0f}%"
                x_text = (f"XInput: {x.rate_hz:.0f} polls/s, "
                          f"{x.change_rate_hz:.0f} changes/s "
                          f"({'OK' if x.connected else 'WAIT'}) | ") if x else ""
                print(
                    f"Capture FPS: {c.actual_fps:6.1f} | "
                    f"Ring loss: {c.dxcam_ring_overwrites} | "
                    f"Cadence gaps: {c.source_cadence_gaps} | "
                    f"GameInput: {g.rate_hz:6.1f} Hz "
                    f"({'OK' if g.connected else 'WAIT'}) | "
                    f"{x_text}"
                    f"RB: {'DOWN' if e.rb_pressed else 'UP'} | "
                    f"Episode: {e.state} | Prebuffer: {e.prebuffer_seconds:.2f}s | "
                    f"Encoder queue: {e.active_encoder_queue} | "
                    f"Overflow: {e.active_encoder_overflows} | "
                    f"Pending queue: {e.pending_encoder_queue} | "
                    f"Storage: {e.active_storage_rate_MBps:.2f} MB/s | "
                    f"NVENC: {gpu_text} | Completed: {e.episodes_completed} | "
                    f"Failed: {e.episodes_failed}",
                    flush=True,
                )
                last_status = now
            if (capture.snapshot_stats().errors
                    or (xinput.snapshot_stats().errors if xinput else gameinput.snapshot_stats().errors)):
                LOGGER.error("A capture worker failed; initiating clean shutdown")
                exit_code = 1
                stop_requested.set()
    except Exception:
        LOGGER.exception("Recorder startup/runtime failure")
        exit_code = 1
    finally:
        LOGGER.info("Shutdown requested; finishing active post-roll")
        manager.begin_shutdown()
        deadline = time.monotonic() + config.episode.post_roll_seconds + 1.0
        while not manager.shutdown_ready() and time.monotonic() < deadline:
            time.sleep(0.02)
        manager.force_close_active()
        if xinput:
            xinput.stop()
        gameinput.stop()
        capture.stop()
        manager.stop_housekeeping()
        LOGGER.info("Flushing encoders and metadata")
        manager.wait_for_writers()
        if metrics_handle is not None:
            metrics_handle.close()
        LOGGER.info("Recorder stopped cleanly")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
