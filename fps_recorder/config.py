from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CaptureConfig:
    device_idx: int = 0
    output_idx: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 60
    resolution_mode: str = "full_frame_gpu_downscale"
    region: tuple[int, int, int, int] | None = None
    center_crop_if_needed: bool = True
    dxcam_buffer_frames: int = 32
    output_color: str = "BGR"
    # DXcam 0.3's Cython color conversion uses OpenMP above this pixel count.
    # On this 4K machine the serial path is much faster and uses less CPU.
    numpy_parallel_threshold_pixels: int = 1_000_000_000


@dataclass(frozen=True)
class GameInputConfig:
    poll_hz: int = 2000
    calibration_samples: int = 128
    calibration_keep_best: int = 24
    reconnect_seconds: float = 1.0


@dataclass(frozen=True)
class XInputConfig:
    enabled: bool = False
    poll_hz: int = 4000
    user_index: int | None = None
    batch_ms: int = 5


@dataclass(frozen=True)
class EpisodeConfig:
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 2.0
    finalization_grace_ms: int = 100
    output_dir: str = "recordings"
    encoder_queue_frames: int = 360
    minimum_free_gib: float = 20.0


@dataclass(frozen=True)
class EncoderConfig:
    ffmpeg: str = "ffmpeg"
    codec: str = "av1_nvenc"
    preset: str = "p5"
    tune: str = "hq"
    rate_control: str = "vbr"
    output_fps: int = 60
    target_bitrate_mbps: float = 12.0
    max_bitrate_mbps: float = 14.0
    cq: int = 29
    buffer_mbps: float = 24.0
    gop_frames: int = 120
    b_frames: int = 3
    lookahead_frames: int = 20
    spatial_aq: bool = True
    temporal_aq: bool = True
    aq_strength: int = 8
    multipass: str = "qres"
    pixel_format: str = "bgr24"


@dataclass(frozen=True)
class StatusConfig:
    interval_seconds: float = 1.0
    gpu_sample_seconds: float = 5.0


@dataclass(frozen=True)
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    gameinput: GameInputConfig = field(default_factory=GameInputConfig)
    xinput: XInputConfig = field(default_factory=XInputConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    status: StatusConfig = field(default_factory=StatusConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        capture_raw = dict(raw.get("capture", {}))
        if capture_raw.get("region") is not None:
            capture_raw["region"] = tuple(int(v) for v in capture_raw["region"])
        episode_raw = dict(raw.get("episode", {}))
        output_dir = Path(episode_raw.get("output_dir", "recordings"))
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
        episode_raw["output_dir"] = str(output_dir.resolve())
        result = cls(
            capture=CaptureConfig(**capture_raw),
            gameinput=GameInputConfig(**raw.get("gameinput", {})),
            xinput=XInputConfig(**raw.get("xinput", {})),
            episode=EpisodeConfig(**episode_raw),
            encoder=EncoderConfig(**raw.get("encoder", {})),
            status=StatusConfig(**raw.get("status", {})),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not 100 <= self.xinput.poll_hz <= 8000:
            raise ValueError("xinput.poll_hz must be between 100 and 8000")
        if self.xinput.user_index not in (None, 0, 1, 2, 3):
            raise ValueError("xinput.user_index must be null or 0..3")
        if not 1 <= self.xinput.batch_ms <= 20:
            raise ValueError("xinput.batch_ms must be between 1 and 20")
        if self.capture.fps not in {60, 120}:
            raise ValueError("capture.fps must be 60 or 120")
        if self.capture.width != 1920 or self.capture.height != 1080:
            raise ValueError("capture resolution must remain 1920x1080")
        color_pairs = {"BGR": "bgr24", "BGRA": "bgra"}
        if self.capture.output_color not in color_pairs:
            raise ValueError("capture.output_color must be BGR or BGRA")
        if self.capture.numpy_parallel_threshold_pixels < 0:
            raise ValueError("capture.numpy_parallel_threshold_pixels must be nonnegative")
        if self.encoder.pixel_format != color_pairs[self.capture.output_color]:
            raise ValueError("encoder.pixel_format does not match capture.output_color")
        if self.capture.resolution_mode not in {
            "full_frame_gpu_downscale",
            "center_crop",
        }:
            raise ValueError("unsupported capture.resolution_mode")
        if self.episode.pre_roll_seconds < 2.0:
            raise ValueError("pre-roll must be at least 2 seconds")
        if self.episode.post_roll_seconds < 2.0:
            raise ValueError("post-roll must be at least 2 seconds")
        if self.encoder.max_bitrate_mbps >= 16.0:
            raise ValueError("max bitrate must remain below the 16 Mbps hard target")
        if self.encoder.output_fps <= 0 or self.capture.fps % self.encoder.output_fps:
            raise ValueError("encoder.output_fps must divide capture.fps")
        if self.episode.encoder_queue_frames < self.encoder.output_fps * 4:
            raise ValueError("encoder queue must cover at least four seconds")
