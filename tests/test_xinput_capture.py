import csv
from dataclasses import replace

import numpy as np

from fps_recorder.config import AppConfig
from fps_recorder.gameinput_capture import BUTTON_RB
from fps_recorder.models import FramePacket
from fps_recorder.writer import write_xinput_csv
from fps_recorder.xinput_capture import NativeState, make_packet
from test_episode_manager import FakeClock, FakeRecorder, pad


def xpad(t, rb=False):
    state = NativeState()
    state.pad.buttons = 0x200 if rb else 0
    return make_packet(state, t - 1000, t, True)


def test_native_mapping_and_separate_timestamp(tmp_path):
    state = NativeState()
    state.number = 42
    state.pad.buttons = 0x200 | 0x1000
    state.pad.lx = -32768
    state.pad.rx = 32767
    state.pad.rt = 255
    row = make_packet(state, 100, 200, True)
    assert row.pressed(BUTTON_RB) and row.lx == -1 and row.rx == 1 and row.rt == 1
    path = tmp_path / "xinput.csv"
    write_xinput_csv(path, [row])
    with path.open(newline="") as handle:
        value = next(csv.DictReader(handle))
    assert value["timestamp_ns"] == "200" and value["poll_before_ns"] == "100"
    assert value["rb"] == "1" and value["a"] == "1"
    assert "gameinput_timestamp_us" not in value


def manager_fixture(monkeypatch, tmp_path):
    import fps_recorder.episode_manager as module
    FakeRecorder.instances.clear()
    monkeypatch.setattr(module, "EpisodeRecorder", FakeRecorder)
    config = AppConfig.load("config.yaml")
    config = replace(config, episode=replace(config.episode,
                     output_dir=str(tmp_path), minimum_free_gib=0))
    clock = FakeClock(10_000_000_000)
    return module.EpisodeManager(config, clock, lambda: {}), clock


def test_dual_trigger_preroll_merge_and_new_episode(monkeypatch, tmp_path):
    manager, clock = manager_fixture(monkeypatch, tmp_path)
    t = clock.value
    try:
        # Startup while held must NOT create an episode.
        manager.on_xinput(xpad(t - 3_000_000_000, True))
        assert not FakeRecorder.instances
        manager.on_xinput(xpad(t - 2_100_000_000))
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for i in range(121):
            stamp = t - 2_000_000_000 + i * 16_666_666
            manager.on_frame(FramePacket(i, stamp, stamp // 100, image))
        manager.on_gamepad(pad(t - 2_050_000_000, False))
        manager.on_gamepad(pad(t, True))  # GI must not trigger.
        assert not FakeRecorder.instances
        manager.on_xinput(xpad(t, True))
        first = FakeRecorder.instances[0]
        assert first.episode_start_ns == t - 2_000_000_000
        assert first.frames[0].timestamp_ns == first.episode_start_ns
        assert len(first.frames) == 121
        assert first.gamepad_rows[0].timestamp_ns < first.episode_start_ns
        assert first.xinput_rows[0].timestamp_ns < first.episode_start_ns
        manager.on_xinput(xpad(t + 100_000_000))
        end = t + 2_100_000_000
        manager.on_xinput(xpad(end, True))  # exact boundary still merges
        assert len(FakeRecorder.instances) == 1
        manager.on_xinput(xpad(end + 100_000_000))
        new_press = end + 2_100_000_001
        manager.on_xinput(xpad(new_press, True))
        assert len(FakeRecorder.instances) == 2
        assert first.closed_at == end + 2_100_000_000
        assert manager.snapshot_stats().state == "RECORDING"
        # Continuous buffers survive the cut; GI remains separate.
        assert len(manager.frame_ring) > 0 and len(manager.xinput_ring) > 0
    finally:
        manager.stop_housekeeping()


def test_disconnect_does_not_retrigger_held_rb(monkeypatch, tmp_path):
    manager, clock = manager_fixture(monkeypatch, tmp_path)
    t = clock.value
    try:
        manager.on_xinput(xpad(t))
        manager.on_xinput(xpad(t + 1, True))
        manager.on_input_disconnect(t + 100)
        manager.on_xinput(xpad(t + 200, True))
        assert len(FakeRecorder.instances) == 1
        assert len(FakeRecorder.instances[0].rb_down_timestamps) == 1
        assert manager._planned_end_ns == t + 100 + 2_000_000_000
        assert FakeRecorder.instances[0].input_disconnect_timestamps == [t + 100]
    finally:
        manager.stop_housekeeping()


def test_fractional_frame_boundary_retains_full_two_seconds(monkeypatch, tmp_path):
    manager, clock = manager_fixture(monkeypatch, tmp_path)
    t = clock.value
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        manager.on_xinput(xpad(t - 2_100_000_000))
        manager.on_gamepad(pad(t - 2_100_000_000, False))
        for i in range(122):
            stamp = t - 2_010_000_000 + i * 16_666_666
            manager.on_frame(FramePacket(i, stamp, stamp // 100, image))
        manager.on_xinput(xpad(t, True))
        recorder = FakeRecorder.instances[0]
        assert t - recorder.frames[0].timestamp_ns >= 2_000_000_000
        assert recorder.xinput_rows[0].timestamp_ns <= recorder.frames[0].timestamp_ns
        assert recorder.gamepad_rows[0].timestamp_ns <= recorder.frames[0].timestamp_ns
    finally:
        manager.stop_housekeeping()
