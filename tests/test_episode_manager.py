from __future__ import annotations

import queue
import time
from dataclasses import replace
from pathlib import Path

from fps_recorder.config import AppConfig
from fps_recorder.gameinput_capture import BUTTON_RB
from fps_recorder.models import GamepadPacket
from fps_recorder.models import FramePacket

import numpy as np


class FakeClock:
    def __init__(self, now_ns: int) -> None:
        self.value = now_ns

    def now_ns(self) -> int:
        return self.value


class FakeRecorder:
    instances = []

    def __init__(
        self,
        *,
        episode_id,
        config,
        episode_start_ns,
        output_root,
        base_metadata,
        on_finished,
    ):
        self.episode_id = episode_id
        self.episode_start_ns = episode_start_ns
        self.rb_down_timestamps = []
        self.rb_up_timestamps = []
        self.gamepad_rows = []
        self.xinput_rows = []
        self.input_disconnect_timestamps = []
        self.frames = []
        self.queue = queue.Queue()
        self.queue_overflow = 0
        self.video_path = Path(output_root) / "missing.mkv"
        self.final_dir = Path(output_root) / f"episode_{episode_id:06d}"
        self.failed_dir = Path(output_root) / f"episode_{episode_id:06d}.failed"
        self.succeeded = True
        self.failure_reason = None
        self.on_finished = on_finished
        self.closed_at = None
        self.final_metadata = None
        self.__class__.instances.append(self)

    def submit_frame(self, frame):
        self.frames.append(frame)
        return True

    def submit_gamepad(self, reading):
        self.gamepad_rows.append(reading)

    def submit_xinput(self, reading):
        self.xinput_rows.append(reading)

    def set_final_metadata(self, values):
        self.final_metadata = values

    def close_async(self, end_ns):
        self.closed_at = end_ns
        self.on_finished(self)

    def join(self, timeout=None):
        return None


def pad(timestamp_ns: int, rb: bool):
    return GamepadPacket(
        timestamp_ns,
        timestamp_ns // 1000,
        BUTTON_RB if rb else 0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_rb_edges_and_post_roll_merge(monkeypatch, tmp_path):
    import fps_recorder.episode_manager as module

    FakeRecorder.instances.clear()
    monkeypatch.setattr(module, "EpisodeRecorder", FakeRecorder)
    config = AppConfig.load("config.yaml")
    config = replace(
        config,
        xinput=replace(config.xinput, enabled=False),
        episode=replace(
            config.episode,
            output_dir=str(tmp_path),
            minimum_free_gib=0,
        ),
    )
    clock = FakeClock(10_000_000_000)
    manager = module.EpisodeManager(config, clock, lambda: {"ok": True})
    try:
        manager.on_gamepad(pad(clock.value, False))
        clock.value += 100
        manager.on_gamepad(pad(clock.value, True))
        manager.on_gamepad(pad(clock.value + 1, True))
        assert len(FakeRecorder.instances) == 1
        recorder = FakeRecorder.instances[0]
        assert len(recorder.rb_down_timestamps) == 1

        clock.value += 500_000_000
        manager.on_gamepad(pad(clock.value, False))
        first_planned_end = clock.value + 2_000_000_000

        clock.value += 1_000_000_000
        manager.on_gamepad(pad(clock.value, True))
        assert len(FakeRecorder.instances) == 1
        assert len(recorder.rb_down_timestamps) == 2

        clock.value += 200_000_000
        manager.on_gamepad(pad(clock.value, False))
        final_end = clock.value + 2_000_000_000
        assert final_end > first_planned_end
        clock.value = final_end + 200_000_000
        deadline = time.monotonic() + 1
        while recorder.closed_at is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert recorder.closed_at == final_end
        assert manager.snapshot_stats().episodes_completed == 1
    finally:
        manager.stop_housekeeping()


def test_encoder_failure_is_not_counted_as_completed(monkeypatch, tmp_path):
    import fps_recorder.episode_manager as module

    FakeRecorder.instances.clear()
    monkeypatch.setattr(module, "EpisodeRecorder", FakeRecorder)
    config = AppConfig.load("config.yaml")
    config = replace(
        config,
        xinput=replace(config.xinput, enabled=False),
        episode=replace(
            config.episode,
            output_dir=str(tmp_path),
            minimum_free_gib=0,
        ),
    )
    clock = FakeClock(10_000_000_000)
    manager = module.EpisodeManager(config, clock, lambda: {"ok": True})
    try:
        manager.on_gamepad(pad(clock.value, False))
        manager.on_gamepad(pad(clock.value + 1, True))
        recorder = FakeRecorder.instances[0]
        recorder.succeeded = False
        recorder.failure_reason = "synthetic encoder failure"
        recorder.on_finished(recorder)

        stats = manager.snapshot_stats()
        assert stats.state == "IDLE"
        assert stats.episodes_completed == 0
        assert stats.episodes_failed == 1
    finally:
        manager.stop_housekeeping()


def test_120hz_capture_selects_every_other_global_frame_for_60fps(monkeypatch, tmp_path):
    import fps_recorder.episode_manager as module

    FakeRecorder.instances.clear()
    monkeypatch.setattr(module, "EpisodeRecorder", FakeRecorder)
    config = AppConfig.load("config.yaml")
    config = replace(
        config,
        xinput=replace(config.xinput, enabled=False),
        capture=replace(config.capture, fps=120),
        episode=replace(config.episode, output_dir=str(tmp_path), minimum_free_gib=0),
    )
    clock = FakeClock(10_000_000_000)
    manager = module.EpisodeManager(config, clock, lambda: {})
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        manager.on_gamepad(pad(clock.value, False))
        for global_id in range(240):
            timestamp = clock.value - 2_000_000_000 + global_id * 8_333_333
            manager.on_frame(FramePacket(global_id, timestamp, timestamp // 100, image))
        manager.on_gamepad(pad(clock.value, True))
        recorder = FakeRecorder.instances[0]
        assert [frame.global_frame_id for frame in recorder.frames] == list(range(0, 240, 2))
        for global_id in range(240, 250):
            timestamp = clock.value + (global_id - 239) * 8_333_333
            manager.on_frame(FramePacket(global_id, timestamp, timestamp // 100, image))
        assert [frame.global_frame_id for frame in recorder.frames[-5:]] == [240, 242, 244, 246, 248]
    finally:
        manager.stop_housekeeping()
