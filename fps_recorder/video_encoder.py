from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .models import FramePacket, GamepadPacket, XInputPacket
from .utils import resolve_executable
from .writer import write_frames_csv, write_gamepad_csv, write_metadata, write_xinput_csv


LOGGER = logging.getLogger(__name__)
_STOP = object()


def build_ffmpeg_command(
    config: AppConfig,
    output_path: Path,
    *,
    input_width: int,
    input_height: int,
) -> list[str]:
    capture = config.capture
    encoder = config.encoder
    command = [
        resolve_executable(encoder.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        encoder.pixel_format,
        "-video_size",
        f"{input_width}x{input_height}",
        "-framerate",
        str(encoder.output_fps),
        "-i",
        "pipe:0",
        "-an",
    ]
    # AV1 NVENC does not accept the YUV444P path FFmpeg may select when fed
    # BGR24 directly.  Pixel-format conversion is mandatory even when the
    # captured surface is already 1920x1080; only scaling is conditional.
    video_filter = "format=nv12"
    if input_width != capture.width or input_height != capture.height:
        video_filter = (
            f"scale={capture.width}:{capture.height}:flags=fast_bilinear,"
            "format=nv12"
        )
    command.extend(["-filter_threads", "16", "-vf", video_filter])
    command.extend(
        [
        "-c:v",
        encoder.codec,
        "-preset",
        encoder.preset,
        "-tune",
        encoder.tune,
        "-rc",
        encoder.rate_control,
        "-b:v",
        f"{encoder.target_bitrate_mbps:g}M",
        "-maxrate",
        f"{encoder.max_bitrate_mbps:g}M",
        "-bufsize",
        f"{encoder.buffer_mbps:g}M",
        "-cq",
        str(encoder.cq),
        "-multipass",
        encoder.multipass,
        "-rc-lookahead",
        str(encoder.lookahead_frames),
        "-spatial-aq",
        "1" if encoder.spatial_aq else "0",
        "-temporal-aq",
        "1" if encoder.temporal_aq else "0",
        "-aq-strength",
        str(encoder.aq_strength),
        "-g",
        str(encoder.gop_frames),
        "-bf",
        str(encoder.b_frames),
        "-fps_mode",
        "cfr",
        "-f",
        "matroska",
        str(output_path),
        ]
    )
    return command


class EpisodeRecorder:
    """One episode's non-blocking video queue and background file writer."""

    def __init__(
        self,
        *,
        episode_id: int,
        config: AppConfig,
        episode_start_ns: int,
        output_root: Path,
        base_metadata: dict[str, Any],
        on_finished: Callable[["EpisodeRecorder"], None] | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.config = config
        self.episode_start_ns = episode_start_ns
        self.episode_end_ns: int | None = None
        self.rb_down_timestamps: list[int] = []
        self.rb_up_timestamps: list[int] = []
        self.gamepad_rows: list[GamepadPacket] = []
        self.xinput_reading_count = 0
        self.xinput_state_change_count = 0
        self.xinput_write_errors: list[str] = []
        self._xinput_queue = queue.Queue(maxsize=65536)
        self._xinput_thread = None
        self.input_disconnect_timestamps: list[int] = []
        self.frame_rows: list[dict[str, int]] = []
        self.queue: queue.Queue[FramePacket | object] = queue.Queue(
            maxsize=config.episode.encoder_queue_frames
        )
        self.queue_overflow = 0
        self.max_queue_depth = 0
        self.encoder_errors: list[str] = []
        self.encoder_return_code: int | None = None
        self.encoder_elapsed_seconds = 0.0
        self.encoder_startup_seconds = 0.0
        self.encoder_pipe_write_seconds = 0.0
        self.encoder_queue_wait_seconds = 0.0
        self.encoder_flush_seconds = 0.0
        self.encoder_queue_depth_at_close = 0
        self.encoder_queue_depth_after_preroll = 0
        self.succeeded = False
        self.failure_reason: str | None = None
        self.video_probe_ok = False
        self._accepting = True
        self._lock = threading.Lock()
        self._base_metadata = base_metadata
        self._final_metadata: dict[str, Any] = {}
        self._on_finished = on_finished
        self.inprogress_dir = output_root / f"episode_{episode_id:06d}.inprogress"
        self.final_dir = output_root / f"episode_{episode_id:06d}"
        self.failed_dir = output_root / f"episode_{episode_id:06d}.failed"
        self.inprogress_dir.mkdir(parents=True, exist_ok=False)
        self.video_path = self.inprogress_dir / "video.mkv"
        if config.xinput.enabled:
            self._xinput_thread = threading.Thread(target=self._write_xinput,
                name=f"XInputWriter-{episode_id:06d}", daemon=True)
            self._xinput_thread.start()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"EpisodeEncoder-{episode_id:06d}",
            daemon=False,
        )
        self._thread.start()

    def submit_frame(self, frame: FramePacket) -> bool:
        with self._lock:
            if not self._accepting:
                return False
        try:
            self.queue.put_nowait(frame)
            self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())
            return True
        except queue.Full:
            self.queue_overflow += 1
            return False

    def submit_gamepad(self, reading: GamepadPacket) -> None:
        with self._lock:
            if self._accepting:
                self.gamepad_rows.append(reading)

    def close_async(self, episode_end_ns: int) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            self.episode_end_ns = int(episode_end_ns)
            self.encoder_queue_depth_at_close = self.queue.qsize()
        # The queue was sized to make this sentinel insertion effectively
        # immediate. If full, wait in a helper, never in a capture callback.
        try:
            self.queue.put_nowait(_STOP)
        except queue.Full:
            threading.Thread(
                target=self.queue.put,
                args=(_STOP,),
                name=f"EpisodeClose-{self.episode_id:06d}",
                daemon=True,
            ).start()

    def submit_xinput(self, reading: XInputPacket) -> None:
        with self._lock:
            if self._accepting:
                try:
                    self._xinput_queue.put_nowait(reading)
                    self.xinput_reading_count += 1
                    self.xinput_state_change_count += int(reading.state_changed)
                except queue.Full:
                    if not self.xinput_write_errors:
                        self.xinput_write_errors.append("XInput disk-writer queue overflow")

    def _write_xinput(self) -> None:
        def rows():
            while True:
                row = self._xinput_queue.get()
                if row is _STOP:
                    return
                yield row
        try:
            write_xinput_csv(self.inprogress_dir / "xinput.csv", rows())
        except Exception as exc:
            self.xinput_write_errors.append(repr(exc))
            LOGGER.exception("XInput disk writer failed")

    def set_final_metadata(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._final_metadata = dict(values)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _worker(self) -> None:
        started = time.perf_counter()
        process: subprocess.Popen[bytes] | None = None
        stderr_lines: list[str] = []
        stderr_thread: threading.Thread | None = None
        try:
            first_item = self.queue.get()
            if first_item is _STOP:
                self.queue.task_done()
                self.encoder_return_code = 0
                return
            assert isinstance(first_item, FramePacket)
            input_height, input_width = first_item.image.shape[:2]
            command = build_ffmpeg_command(
                self.config,
                self.video_path,
                input_width=input_width,
                input_height=input_height,
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.encoder_startup_seconds = time.perf_counter() - started

            def read_stderr() -> None:
                assert process is not None and process.stderr is not None
                for raw in iter(process.stderr.readline, b""):
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        stderr_lines.append(line)
                        del stderr_lines[:-100]

            stderr_thread = threading.Thread(
                target=read_stderr, name="FFmpegStderr", daemon=True
            )
            stderr_thread.start()
            assert process.stdin is not None
            frame_id = 0
            pending_item: FramePacket | object = first_item
            while True:
                item = pending_item
                try:
                    if item is _STOP:
                        break
                    assert isinstance(item, FramePacket)
                    write_started = time.perf_counter()
                    process.stdin.write(memoryview(item.image).cast("B"))
                    self.encoder_pipe_write_seconds += time.perf_counter() - write_started
                    self.frame_rows.append(
                        {
                            "frame_id": frame_id,
                            "source_present_ns": item.timestamp_ns,
                            "video_pts_ns": frame_id * 1_000_000_000 // self.config.encoder.output_fps,
                            "is_duplicate": int(item.is_duplicate),
                            "dxgi_present_ticks": item.dxgi_present_ticks,
                            "global_frame_id": item.global_frame_id,
                        }
                    )
                    frame_id += 1
                finally:
                    self.queue.task_done()
                wait_started = time.perf_counter()
                pending_item = self.queue.get()
                self.encoder_queue_wait_seconds += time.perf_counter() - wait_started
            flush_started = time.perf_counter()
            process.stdin.close()
            self.encoder_return_code = process.wait(timeout=60.0)
            self.encoder_flush_seconds = time.perf_counter() - flush_started
            if stderr_thread:
                stderr_thread.join(timeout=1.0)
            if self.encoder_return_code:
                self.encoder_errors.extend(stderr_lines)
        except Exception as exc:
            with self._lock:
                self._accepting = False
            LOGGER.exception("Episode %d encoder failed", self.episode_id)
            self.encoder_errors.append(repr(exc))
            if process and process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)
            if stderr_thread:
                stderr_thread.join(timeout=1.0)
            self.encoder_errors.extend(stderr_lines)
            self.encoder_return_code = process.returncode if process else -1
        finally:
            self.encoder_elapsed_seconds = time.perf_counter() - started
            try:
                self._finalize_files()
            finally:
                if self._on_finished:
                    self._on_finished(self)

    def _finalize_files(self) -> None:
        with self._lock:
            self._accepting = False
        if self._xinput_thread:
            if self._xinput_thread.is_alive():
                try:
                    self._xinput_queue.put(_STOP, timeout=5)
                except queue.Full:
                    self.xinput_write_errors.append("XInput writer failed to drain")
                self._xinput_thread.join(timeout=30)
                if self._xinput_thread.is_alive():
                    self.xinput_write_errors.append("XInput writer did not finish")
        file_size = self.video_path.stat().st_size if self.video_path.exists() else 0
        self.video_probe_ok = file_size > 0 and self._probe_video()
        self.succeeded = (
            self.encoder_return_code == 0
            and file_size > 0
            and len(self.frame_rows) > 0
            and self.video_probe_ok
            and self.queue_overflow == 0
            and not self.xinput_write_errors
        )
        if not self.succeeded:
            if self.xinput_write_errors:
                self.failure_reason = self.xinput_write_errors[0]
            elif self.queue_overflow:
                self.failure_reason = f"Encoder queue overflowed by {self.queue_overflow} frames"
            elif self.encoder_return_code not in (None, 0):
                self.failure_reason = (
                    f"FFmpeg exited with code {self.encoder_return_code}"
                )
            elif file_size == 0:
                self.failure_reason = "FFmpeg produced an empty video file"
            elif not self.video_probe_ok:
                self.failure_reason = "ffprobe could not read a video stream"
            else:
                self.failure_reason = "No video frames were encoded"
            if self.failure_reason not in self.encoder_errors:
                self.encoder_errors.append(self.failure_reason)
        if len(self.frame_rows) >= 2:
            media_span_ns = (
                self.frame_rows[-1]["source_present_ns"]
                - self.frame_rows[0]["source_present_ns"]
            )
        else:
            media_span_ns = 0
        duration_seconds = media_span_ns / 1_000_000_000
        bitrate_mbps = (
            file_size * 8 / duration_seconds / 1_000_000
            if duration_seconds > 0
            else 0.0
        )
        write_frames_csv(self.inprogress_dir / "frames.csv", self.frame_rows)
        write_gamepad_csv(self.inprogress_dir / "gamepad.csv", self.gamepad_rows)
        metadata = {
            **self._base_metadata,
            "data_format_version": 3 if self.config.xinput.enabled else 2,
            "rb_trigger_source": "xinput" if self.config.xinput.enabled else "gameinput",
            "input_disconnect_timestamps_ns": self.input_disconnect_timestamps,
            "xinput_reading_count": self.xinput_reading_count,
            "xinput_state_change_count": self.xinput_state_change_count,
            "xinput_write_errors": self.xinput_write_errors,
            "xinput_timestamp_source": "QPC after XInputGetState (host observation, not device event)",
            "xinput_config": asdict(self.config.xinput),
            "alignment_rule": "latest input timestamp_ns <= frame source_present_ns; no future samples",
            "input_quality_warning": bool(self.input_disconnect_timestamps
                or self.xinput_write_errors
                or (self._final_metadata.get("xinput_stats") or {}).get("errors")
                or (self._final_metadata.get("gameinput_stats") or {}).get("errors")),
            "recommended_action_stream": "xinput.csv" if self.config.xinput.enabled else "gamepad.csv",
            "actual_pre_roll_seconds": (
                (self.rb_down_timestamps[0] - self.frame_rows[0]["source_present_ns"]) / 1e9
                if self.rb_down_timestamps and self.frame_rows else None
            ),
            "actual_post_roll_seconds": (
                (self.frame_rows[-1]["source_present_ns"] - self.rb_up_timestamps[-1]) / 1e9
                if self.rb_up_timestamps and self.frame_rows else None
            ),
            "final_runtime": self._final_metadata,
            "episode_id": self.episode_id,
            "episode_start_ns": self.episode_start_ns,
            "rb_down_timestamps_ns": self.rb_down_timestamps,
            "rb_up_timestamps_ns": self.rb_up_timestamps,
            "episode_end_ns": self.episode_end_ns,
            "actual_first_frame_ns": (
                self.frame_rows[0]["source_present_ns"] if self.frame_rows else None
            ),
            "actual_last_frame_ns": (
                self.frame_rows[-1]["source_present_ns"] if self.frame_rows else None
            ),
            "frame_count": len(self.frame_rows),
            "duplicate_frame_count": sum(row["is_duplicate"] for row in self.frame_rows),
            "gameinput_reading_count": len(self.gamepad_rows),
            "encoder_queue_overflow": self.queue_overflow,
            "encoder_queue_max_depth": self.max_queue_depth,
            "encoder_queue_depth_after_preroll": self.encoder_queue_depth_after_preroll,
            "encoder_queue_depth_at_close": self.encoder_queue_depth_at_close,
            "encoder_startup_seconds": self.encoder_startup_seconds,
            "encoder_pipe_write_seconds": self.encoder_pipe_write_seconds,
            "encoder_pipe_write_fps": (
                len(self.frame_rows) / self.encoder_pipe_write_seconds
                if self.encoder_pipe_write_seconds > 0
                else 0.0
            ),
            "encoder_queue_wait_seconds": self.encoder_queue_wait_seconds,
            "encoder_flush_seconds": self.encoder_flush_seconds,
            "encoder_elapsed_seconds": self.encoder_elapsed_seconds,
            "encoder_fps": (
                len(self.frame_rows) / self.encoder_elapsed_seconds
                if self.encoder_elapsed_seconds > 0
                else 0.0
            ),
            "encoder_return_code": self.encoder_return_code,
            "encoder_errors": self.encoder_errors,
            "recording_status": "complete" if self.succeeded else "failed",
            "failure_reason": self.failure_reason,
            "video_probe_ok": self.video_probe_ok,
            "video_file_bytes": file_size,
            "actual_average_bitrate_mbps": bitrate_mbps,
            "actual_storage_rate_MBps": bitrate_mbps / 8,
            "actual_media_span_seconds": duration_seconds,
            "encoded_playback_duration_seconds": len(self.frame_rows) / self.config.encoder.output_fps,
            "versions": {
                "dxcam": importlib.metadata.version("dxcam"),
            },
            "encoder": asdict(self.config.encoder),
            "capture": asdict(self.config.capture),
            "episode": {
                "pre_roll_seconds": self.config.episode.pre_roll_seconds,
                "post_roll_seconds": self.config.episode.post_roll_seconds,
            },
        }
        write_metadata(self.inprogress_dir / "metadata.json", metadata)
        destination = self.final_dir if self.succeeded else self.failed_dir
        if destination.exists():
            raise FileExistsError(destination)
        self.inprogress_dir.rename(destination)

    def _probe_video(self) -> bool:
        try:
            ffmpeg_path = Path(resolve_executable(self.config.encoder.ffmpeg))
            sibling_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
            sibling = ffmpeg_path.with_name(sibling_name)
            ffprobe = str(sibling) if sibling.is_file() else resolve_executable("ffprobe")
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_packets",
                    "-show_entries",
                    "stream=codec_name,nb_read_packets",
                    "-of",
                    "json",
                    str(self.video_path),
                ],
                capture_output=True,
                text=True,
                timeout=15.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0 and result.stdout.strip():
                probe = json.loads(result.stdout)
                streams = probe.get("streams", [])
                if streams:
                    packets = int(streams[0].get("nb_read_packets", -1))
                    if packets == len(self.frame_rows):
                        return True
                    self.encoder_errors.append(
                        f"video packets {packets} != frames.csv rows {len(self.frame_rows)}"
                    )
                    return False
            detail = result.stderr.strip() or "no video stream reported"
            self.encoder_errors.append(f"ffprobe validation failed: {detail}")
        except Exception as exc:
            self.encoder_errors.append(f"ffprobe validation failed: {exc!r}")
        return False
