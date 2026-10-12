import csv
import json
import queue
import threading
from pathlib import Path
from dataclasses import replace

import numpy as np

from fps_recorder.config import AppConfig
from fps_recorder.video_encoder import EpisodeRecorder
from fps_recorder.models import FramePacket
from test_xinput_capture import xpad
from test_episode_manager import pad


def test_background_streaming_and_metadata(tmp_path, monkeypatch):
    # Exercise real CSV writer/finalizer, without launching FFmpeg or injecting
    # any controller output. This fixture is not a training recording.
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.config = AppConfig.load("config.yaml")
    recorder.episode_id = 1
    recorder.episode_start_ns = 10
    recorder.episode_end_ns = 300
    recorder.rb_down_timestamps = [100]
    recorder.rb_up_timestamps = [200]
    recorder.gamepad_rows = []
    recorder.frame_rows = [{"frame_id": 0, "source_present_ns": 10,
        "video_pts_ns": 0, "is_duplicate": 0, "dxgi_present_ticks": 1,
        "global_frame_id": 1}]
    recorder._lock = threading.Lock()
    recorder._accepting = True
    recorder._xinput_queue = queue.Queue(maxsize=65536)
    recorder.xinput_reading_count = 0
    recorder.xinput_state_change_count = 0
    recorder.xinput_write_errors = []
    recorder.input_disconnect_timestamps = []
    recorder.inprogress_dir = tmp_path / "fixture.inprogress"
    recorder.inprogress_dir.mkdir()
    recorder.final_dir = tmp_path / "fixture"
    recorder.failed_dir = tmp_path / "fixture.failed"
    recorder.video_path = recorder.inprogress_dir / "video.mkv"
    recorder.video_path.write_bytes(b"synthetic probe fixture")
    recorder._base_metadata = {}
    recorder._final_metadata = {}
    for name in ("queue_overflow", "max_queue_depth", "encoder_queue_depth_after_preroll",
                 "encoder_queue_depth_at_close", "encoder_startup_seconds",
                 "encoder_pipe_write_seconds", "encoder_queue_wait_seconds",
                 "encoder_flush_seconds", "encoder_elapsed_seconds", "encoder_return_code"):
        setattr(recorder, name, 0)
    recorder.encoder_errors = []
    recorder.failure_reason = None
    monkeypatch.setattr(recorder, "_probe_video", lambda: True)
    recorder._xinput_thread = threading.Thread(target=recorder._write_xinput)
    recorder._xinput_thread.start()
    for i in range(10000):
        recorder.submit_xinput(xpad(i + 10, i % 2 == 0))
    recorder._finalize_files()
    assert recorder.succeeded and not recorder._xinput_thread.is_alive()
    with (recorder.final_dir / "xinput.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10000
    assert rows[0]["timestamp_ns"] == "10"
    assert rows[-1]["timestamp_ns"] == "10009"
    meta = json.loads((recorder.final_dir / "metadata.json").read_text())
    assert meta["data_format_version"] == 3
    assert meta["xinput_reading_count"] == 10000
    assert meta["rb_trigger_source"] == "xinput"
    assert not hasattr(recorder, "xinput_rows")


def test_actual_nvenc_dual_files(tmp_path):
    """Local hardware integration: synthetic frames, never gameplay training data."""
    config = AppConfig.load("config.yaml")
    config = replace(config, episode=replace(config.episode, output_dir=str(tmp_path)))
    recorder = EpisodeRecorder(episode_id=1, config=config,
        episode_start_ns=1_000_000_000, output_root=tmp_path,
        base_metadata={"validation_only": True})
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for i in range(32):
        stamp = 1_000_000_000 + i * 1_000_000_000 // 60
        assert recorder.submit_frame(FramePacket(i, stamp, stamp // 100, image))
        recorder.submit_gamepad(pad(stamp, False))
        for j in range(4):
            recorder.submit_xinput(xpad(stamp + j * 250_000))
    recorder.close_async(stamp + 1_000_000)
    recorder.join(timeout=30)
    assert not recorder.is_alive
    assert recorder.succeeded, recorder.failure_reason
    meta = json.loads((recorder.final_dir / "metadata.json").read_text())
    assert meta["frame_count"] == 32
    assert meta["gameinput_reading_count"] == 32
    assert meta["xinput_reading_count"] == 128
    for name, count in (("frames.csv", 32), ("gamepad.csv", 32), ("xinput.csv", 128)):
        with (recorder.final_dir / name).open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == count
