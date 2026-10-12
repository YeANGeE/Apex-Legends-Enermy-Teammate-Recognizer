from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .clock import QpcClock
from .config import AppConfig
from .gameinput_capture import BUTTON_RB
from .models import FramePacket, GamepadPacket, XInputPacket
from .ring_buffer import TimestampRing
from .video_encoder import EpisodeRecorder


LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeManagerStats:
    state: str = "IDLE"
    episodes_started: int = 0
    episodes_completed: int = 0
    episodes_failed: int = 0
    rejected_low_disk: int = 0
    rb_pressed: bool = False
    prebuffer_seconds: float = 0.0
    active_encoder_queue: int = 0
    active_encoder_overflows: int = 0
    pending_encoders: int = 0
    pending_encoder_queue: int = 0
    active_storage_rate_MBps: float = 0.0


class EpisodeManager:
    def __init__(
        self,
        config: AppConfig,
        clock: QpcClock,
        metadata_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.config = config
        self.clock = clock
        self.metadata_provider = metadata_provider
        pre_roll_ns = round(config.episode.pre_roll_seconds * 1_000_000_000)
        # One extra capture interval prevents boundary rounding from producing
        # a buffer microscopically shorter than the requested two seconds.
        self.frame_ring = TimestampRing[FramePacket](
            pre_roll_ns + round(2_000_000_000 / config.capture.fps),
            lambda row: row.timestamp_ns,
            max_items=round(
                config.capture.fps * (config.episode.pre_roll_seconds + 0.25)
            ),
        )
        self.gamepad_ring = TimestampRing[GamepadPacket](
            pre_roll_ns + 250_000_000,
            lambda row: row.timestamp_ns,
            max_items=round(
                config.gameinput.poll_hz * (config.episode.pre_roll_seconds + 0.5)
            ),
        )
        self.xinput_ring = TimestampRing[XInputPacket](
            pre_roll_ns + 250_000_000,
            lambda row: row.timestamp_ns,
            max_items=round(config.xinput.poll_hz *
                            (config.episode.pre_roll_seconds + 0.5)),
        )
        self._pre_roll_ns = pre_roll_ns
        self._frame_stride = config.capture.fps // config.encoder.output_fps
        self._post_roll_ns = round(
            config.episode.post_roll_seconds * 1_000_000_000
        )
        self._grace_ns = config.episode.finalization_grace_ms * 1_000_000
        self._lock = threading.RLock()
        self._active: EpisodeRecorder | None = None
        self._planned_end_ns: int | None = None
        self._last_rb: bool | None = None
        self._accept_new = True
        self._pending: list[EpisodeRecorder] = []
        self._completed = 0
        self._started = 0
        self._failed = 0
        self._last_encoder_error: str | None = None
        self._rejected_low_disk = 0
        self._latest_frame_ns = 0
        self._latest_gamepad_ns = 0
        output_root = Path(config.episode.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self._next_episode_id = self._discover_next_id()
        self._stop_housekeeping = threading.Event()
        self._housekeeping = threading.Thread(
            target=self._housekeeping_loop,
            name="EpisodeHousekeeping",
            daemon=True,
        )
        self._housekeeping.start()

    def on_frame(self, frame: FramePacket) -> None:
        with self._lock:
            self.frame_ring.append(frame)
            self._latest_frame_ns = frame.timestamp_ns
            active = self._active
            if active is None:
                return
            if (frame.global_frame_id % self._frame_stride == 0
                    and (self._planned_end_ns is None or frame.timestamp_ns <= self._planned_end_ns)):
                active.submit_frame(frame)

    def on_gamepad(self, reading: GamepadPacket) -> None:
        with self._lock:
            self.gamepad_ring.append(reading)
            self._latest_gamepad_ns = reading.timestamp_ns
            if self.config.xinput.enabled:
                if self._active is not None and (
                    self._planned_end_ns is None or reading.timestamp_ns <= self._planned_end_ns
                ):
                    self._active.submit_gamepad(reading)
                return
            self._handle_trigger_locked(reading)

    def on_xinput(self, reading: XInputPacket) -> None:
        with self._lock:
            self.xinput_ring.append(reading)
            self._handle_trigger_locked(reading)

    def on_input_disconnect(self, timestamp_ns: int) -> None:
        with self._lock:
            if self._active is not None:
                self._active.input_disconnect_timestamps.append(timestamp_ns)
                if self._planned_end_ns is None:
                    self._planned_end_ns = timestamp_ns + self._post_roll_ns
            # Reconnected held RB is a level, not a new press.
            self._last_rb = None

    def _handle_trigger_locked(self, reading: GamepadPacket | XInputPacket) -> None:
        def submit(row):
            if isinstance(row, XInputPacket):
                self._active.submit_xinput(row)
            else:
                self._active.submit_gamepad(row)

        rb = reading.pressed(BUTTON_RB)
        if self._last_rb is None:
            # Initial/reconnected held RB is a level, not a rising edge.
            self._last_rb = rb
            if self._active is not None and (
                self._planned_end_ns is None or reading.timestamp_ns <= self._planned_end_ns
            ):
                submit(reading)
            return
        rising = rb and not self._last_rb
        falling = not rb and self._last_rb
        self._last_rb = rb

        if (rising and self._active is not None
                and self._planned_end_ns is not None
                and reading.timestamp_ns > self._planned_end_ns):
            self._finish_active_locked(self._planned_end_ns)

        if rising:
            if self._active is None:
                if not self._accept_new:
                    return
                self._start_episode_locked(reading)
                return
            if not self._accept_new:
                if reading.timestamp_ns <= (self._planned_end_ns or reading.timestamp_ns):
                    submit(reading)
                return
            self._active.rb_down_timestamps.append(reading.timestamp_ns)
            self._planned_end_ns = None

        if self._active is None:
            return
        if self._planned_end_ns is None or reading.timestamp_ns <= self._planned_end_ns:
            submit(reading)
        if falling:
            self._active.rb_up_timestamps.append(reading.timestamp_ns)
            self._planned_end_ns = reading.timestamp_ns + self._post_roll_ns

    def begin_shutdown(self) -> None:
        """Stop new episodes while allowing the active one a final post-roll."""
        with self._lock:
            self._accept_new = False
            if self._active is not None and self._planned_end_ns is None:
                self._planned_end_ns = self.clock.now_ns() + self._post_roll_ns

    def shutdown_ready(self) -> bool:
        with self._lock:
            return self._active is None

    def force_close_active(self) -> None:
        with self._lock:
            if self._active is not None:
                end_ns = min(
                    self._planned_end_ns or self.clock.now_ns(),
                    self._latest_frame_ns or self.clock.now_ns(),
                )
                self._finish_active_locked(end_ns)

    def stop_housekeeping(self) -> None:
        self._stop_housekeeping.set()
        self._housekeeping.join(timeout=2.0)

    def wait_for_writers(self) -> None:
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return
            for recorder in pending:
                recorder.join(timeout=0.2)

    def snapshot_stats(self) -> EpisodeManagerStats:
        with self._lock:
            active = self._active
            return EpisodeManagerStats(
                state=(
                    "IDLE"
                    if active is None
                    else ("POST_ROLL" if self._planned_end_ns else "RECORDING")
                ),
                episodes_started=self._started,
                episodes_completed=self._completed,
                episodes_failed=self._failed,
                rejected_low_disk=self._rejected_low_disk,
                rb_pressed=bool(self._last_rb),
                prebuffer_seconds=self.frame_ring.span_ns() / 1_000_000_000,
                active_encoder_queue=active.queue.qsize() if active else 0,
                active_encoder_overflows=active.queue_overflow if active else 0,
                pending_encoders=len(self._pending),
                pending_encoder_queue=sum(recorder.queue.qsize() for recorder in self._pending),
                active_storage_rate_MBps=self._active_storage_rate_locked(),
            )

    def _active_storage_rate_locked(self) -> float:
        active = self._active
        if active is None or not active.video_path.exists():
            return 0.0
        elapsed_ns = max(0, self.clock.now_ns() - active.episode_start_ns)
        if elapsed_ns <= 0:
            return 0.0
        return active.video_path.stat().st_size / (elapsed_ns / 1_000_000_000) / 1_000_000

    def _start_episode_locked(self, trigger: GamepadPacket | XInputPacket) -> None:
        free_gib = shutil.disk_usage(self.output_root).free / (1024**3)
        if free_gib < self.config.episode.minimum_free_gib:
            self._rejected_low_disk += 1
            LOGGER.error(
                "Episode rejected: %.1f GiB free, minimum is %.1f GiB",
                free_gib,
                self.config.episode.minimum_free_gib,
            )
            return
        episode_start_ns = trigger.timestamp_ns - self._pre_roll_ns
        recorder = EpisodeRecorder(
            episode_id=self._next_episode_id,
            config=self.config,
            episode_start_ns=episode_start_ns,
            output_root=self.output_root,
            base_metadata=self.metadata_provider(),
            on_finished=self._on_finished,
        )
        self._next_episode_id += 1
        self._started += 1
        # Publish the recorder before feeding pre-roll. FFmpeg can fail as soon
        # as the first frame arrives, and its callback must be able to detach
        # the active recorder immediately.
        self._active = recorder
        self._planned_end_ns = None
        recorder.rb_down_timestamps.append(trigger.timestamp_ns)
        # Include the last stored frame at/before the nominal start, so a
        # fractional frame interval does not cut pre-roll below two seconds.
        candidates = [frame for frame in self.frame_ring.snapshot_since(
            episode_start_ns - round(2_000_000_000 / self.config.capture.fps))
            if frame.global_frame_id % self._frame_stride == 0]
        preceding = [frame for frame in candidates if frame.timestamp_ns < episode_start_ns]
        frames = preceding[-1:] + [frame for frame in candidates
                                  if frame.timestamp_ns >= episode_start_ns]
        for frame in frames:
            recorder.submit_frame(frame)
        recorder.encoder_queue_depth_after_preroll = recorder.queue.qsize()
        input_start_ns = frames[0].timestamp_ns if frames else episode_start_ns
        for reading in self.gamepad_ring.snapshot_with_predecessor(input_start_ns):
            recorder.submit_gamepad(reading)
        if self.config.xinput.enabled:
            for reading in self.xinput_ring.snapshot_with_predecessor(input_start_ns):
                recorder.submit_xinput(reading)
        LOGGER.info(
            "Episode %06d started with %.3f s video pre-roll",
            recorder.episode_id,
            self.frame_ring.span_ns() / 1_000_000_000,
        )

    def _finish_active_locked(self, end_ns: int) -> None:
        recorder = self._active
        if recorder is None:
            return
        self._active = None
        self._planned_end_ns = None
        self._pending.append(recorder)
        recorder.set_final_metadata(self.metadata_provider())
        recorder.close_async(end_ns)
        LOGGER.info("Episode %06d queued for finalization", recorder.episode_id)

    def _on_finished(self, recorder: EpisodeRecorder) -> None:
        with self._lock:
            if self._active is recorder:
                self._active = None
                self._planned_end_ns = None
            if recorder in self._pending:
                self._pending.remove(recorder)
            if recorder.succeeded:
                self._completed += 1
            else:
                self._failed += 1
                self._last_encoder_error = recorder.failure_reason
        if recorder.succeeded:
            LOGGER.info(
                "Episode %06d saved to %s", recorder.episode_id, recorder.final_dir
            )
        else:
            LOGGER.error(
                "Episode %06d failed: %s (diagnostics: %s)",
                recorder.episode_id,
                recorder.failure_reason,
                recorder.failed_dir,
            )

    def _housekeeping_loop(self) -> None:
        while not self._stop_housekeeping.wait(0.02):
            with self._lock:
                if self._active is None or self._planned_end_ns is None:
                    continue
                now_ns = self.clock.now_ns()
                if now_ns < self._planned_end_ns + self._grace_ns:
                    continue
                # Native source timestamps may lag wall QPC during a stalled or
                # minimized display. The grace window bounds that wait without
                # ever stopping the rolling capture devices.
                self._finish_active_locked(self._planned_end_ns)

    def _discover_next_id(self) -> int:
        highest = 0
        for path in self.output_root.glob("episode_*"):
            stem = path.name.removesuffix(".inprogress").removesuffix(".failed")
            try:
                highest = max(highest, int(stem.split("_")[-1]))
            except ValueError:
                continue
        return highest + 1
