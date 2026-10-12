from pathlib import Path
from dataclasses import replace

from fps_recorder.config import AppConfig
from fps_recorder.video_encoder import build_ffmpeg_command


def _video_filter(command: list[str]) -> str:
    return command[command.index("-vf") + 1]


def test_native_1080p_always_converts_to_nv12():
    config = AppConfig.load("config.yaml")
    command = build_ffmpeg_command(
        config, Path("video.mkv"), input_width=1920, input_height=1080
    )

    assert _video_filter(command) == "format=nv12"
    assert command[command.index("-framerate") + 1] == "60"


def test_4k_input_scales_and_converts_to_nv12():
    config = AppConfig.load("config.yaml")
    command = build_ffmpeg_command(
        config, Path("video.mkv"), input_width=3840, input_height=2160
    )

    assert _video_filter(command) == (
        "scale=1920:1080:flags=fast_bilinear,format=nv12"
    )


def test_capture_60_override_still_writes_60_fps_video():
    config = AppConfig.load("config.yaml")
    config = replace(config, capture=replace(config.capture, fps=60))
    config.validate()
    command = build_ffmpeg_command(
        config, Path("video.mkv"), input_width=3840, input_height=2160
    )
    assert command[command.index("-framerate") + 1] == "60"
