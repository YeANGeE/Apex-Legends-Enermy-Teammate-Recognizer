from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .clock import QpcClock
from .config import CaptureConfig
from .models import FramePacket


LOGGER = logging.getLogger(__name__)


@dataclass
class CaptureStats:
    frames: int = 0
    actual_fps: float = 0.0
    dxcam_ring_overwrites: int = 0
    source_cadence_gaps: int = 0
    duplicate_timestamps: int = 0
    errors: int = 0
    latest_timestamp_ns: int = 0
    realtime_skipped_frames: int = 0
    duplicate_copies_avoided: int = 0


def select_ring_sequence(total, previous, capacity, latest_only=False):
    """Separate physical overwrite from intentional realtime backlog dropping."""
    overwritten=max(0,total-previous-capacity)
    available=previous+overwritten
    skipped=max(0,total-available-1) if latest_only else 0
    return available+skipped,overwritten,skipped


def copy_capture_image(image, preview_size=None, output_bgr=False):
    """Owned output; resize under the ring lock to avoid a full-size duplicate."""
    if preview_size is not None and image.shape[:2] != (preview_size[1],preview_size[0]):
        import cv2
        result=cv2.resize(image,preview_size,interpolation=cv2.INTER_AREA)
    else:
        result=np.array(image,copy=True)
    if output_bgr:
        import cv2
        result=cv2.cvtColor(result,cv2.COLOR_BGRA2BGR)
    return result


class DXcamCapture:
    """DXcam producer plus loss-detecting sequential ring consumer.

    DXcam 0.3 exposes raw LastPresentTime ticks, but its public consumer API
    returns only the newest frame. This reader drains the version-pinned 0.3
    ring in sequence and explicitly counts overwrites instead of hiding them.
    """

    def __init__(
        self,
        config: CaptureConfig,
        clock: QpcClock,
        on_frame: Callable[[FramePacket], None],
        output_fps: int | None = None,
        latest_only: bool = False,
        measure_latency: bool = False,
        downscale_for_preview: bool = False,
    ) -> None:
        self.config = config
        self.clock = clock
        self.on_frame = on_frame
        self.latest_only = latest_only
        self.measure_latency = measure_latency
        if downscale_for_preview and not latest_only:
            raise ValueError('Preview downscale requires realtime latest-only mode')
        self.preview_size=(config.width,config.height) if downscale_for_preview else None
        self.preview_bgr=downscale_for_preview and config.output_color=='BGRA'
        self._frame_stride = config.fps // (output_fps or config.fps)
        self.camera = None
        self.stats = CaptureStats()
        self._stats_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import dxcam
        try:
            from dxcam.processor import _numpy_kernels
        except ImportError:
            LOGGER.warning("DXcam NumPy extension is unavailable; using its cv2 fallback")
        else:
            _numpy_kernels.set_parallel_pixels_threshold(
                self.config.numpy_parallel_threshold_pixels
            )
            LOGGER.info(
                "DXcam NumPy parallel threshold: %d pixels",
                self.config.numpy_parallel_threshold_pixels,
            )

        if self._thread and self._thread.is_alive():
            return
        self.camera = dxcam.create(
            device_idx=self.config.device_idx,
            output_idx=self.config.output_idx,
            output_color=self.config.output_color,
            max_buffer_len=self.config.dxcam_buffer_frames,
            backend="dxgi",
            processor_backend="numpy",
        )
        region = self.config.region
        if (
            region is None
            and self.config.resolution_mode == "center_crop"
            and (
            self.camera.width != self.config.width
            or self.camera.height != self.config.height
            )
        ):
            if (
                self.config.center_crop_if_needed
                and self.camera.width >= self.config.width
                and self.camera.height >= self.config.height
            ):
                left = (self.camera.width - self.config.width) // 2
                top = (self.camera.height - self.config.height) // 2
                region = (
                    left,
                    top,
                    left + self.config.width,
                    top + self.config.height,
                )
                LOGGER.warning(
                    "DXGI output is %dx%d; using centered 1080p crop %s. "
                    "Use a native 1920x1080 output to retain the full HUD.",
                    self.camera.width,
                    self.camera.height,
                    region,
                )
            else:
                surface = f"{self.camera.width}x{self.camera.height}"
                self.camera.release()
                self.camera = None
                raise RuntimeError(
                    "DXGI output cannot provide the requested 1920x1080 frame: "
                    f"surface={surface}"
                )
        selected_width = (region[2] - region[0]) if region else self.camera.width
        selected_height = (region[3] - region[1]) if region else self.camera.height
        needs_scale = (
            selected_width != self.config.width
            or selected_height != self.config.height
        )
        if needs_scale and self.config.resolution_mode != "full_frame_gpu_downscale":
            self.camera.release()
            self.camera = None
            raise RuntimeError(
                f"Selected DXGI region is {selected_width}x{selected_height}; "
                "it must be exactly 1920x1080."
            )
        if needs_scale and (
            selected_width < self.config.width or selected_height < self.config.height
        ):
            self.camera.release()
            self.camera = None
            raise RuntimeError("DXGI surface is smaller than the requested output")
        self.source_width = selected_width
        self.source_height = selected_height
        self.active_region = region or (
            0,
            0,
            self.camera.width,
            self.camera.height,
        )
        if needs_scale:
            LOGGER.info(
                "Capturing full %dx%d surface; encoder worker will scale to %dx%d",
                selected_width,
                selected_height,
                self.config.width,
                self.config.height,
            )
        self._stop.clear()
        self.camera.start(
            region=region,
            target_fps=self.config.fps,
            # Keep a 120 Hz capture timeline even when the desktop/game is not
            # presenting new surfaces. DXcam duplicates the previous image in
            # this case and deliberately preserves its LastPresentTime; we
            # count those duplicates rather than inventing a native timestamp.
            video_mode=True,
        )
        self._thread = threading.Thread(
            target=self._drain_loop, name="DXcamRingDrain", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        camera = self.camera
        if camera is not None:
            # Wakes the private frame event through DXcam.stop().
            camera.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if camera is not None:
            camera.release()
        self.camera = None

    def snapshot_stats(self) -> CaptureStats:
        with self._stats_lock:
            return CaptureStats(**vars(self.stats))

    def _drain_loop(self) -> None:
        assert self.camera is not None
        camera = self.camera
        runtime = camera._DXCamera__capture_runtime
        lock = camera._DXCamera__lock
        frame_event = camera._DXCamera__frame_available
        last_total = 0
        last_timestamp_ns: int | None = None
        last_selected_timestamp_ns: int | None = None
        wall_started = time.perf_counter()
        target_period_ns = round(1_000_000_000 / self.config.fps)
        try:
            while not self._stop.is_set():
                frame_event.wait(timeout=0.1)
                frame_event.clear()
                while not self._stop.is_set():
                    with lock:
                        total = int(runtime.frame_count)
                        if total < last_total:
                            LOGGER.warning("DXcam ring was rebuilt; restarting sequence")
                            last_total = 0
                        pending = total - last_total
                        if pending <= 0 or runtime.frame_buffer is None:
                            break
                        sequence,lost,skipped=select_ring_sequence(total,last_total,runtime.max_buffer_len,self.latest_only)
                        last_total=sequence
                        with self._stats_lock:
                            self.stats.dxcam_ring_overwrites += lost
                            self.stats.realtime_skipped_frames += skipped
                            self.stats.frames += skipped
                        index = sequence % runtime.max_buffer_len
                        keep_frame = sequence % self._frame_stride == 0
                        present_ticks = int(runtime.frame_time_ticks[index])
                        timestamp_ns = self.clock.ticks_to_ns(present_ticks)
                        if self.latest_only and timestamp_ns == last_selected_timestamp_ns:
                            keep_frame=False
                            with self._stats_lock:
                                self.stats.duplicate_copies_avoided += 1
                        copy_start_ns = self.clock.now_ns() if self.measure_latency and keep_frame else None
                        frame = (
                            copy_capture_image(runtime.frame_buffer[index],self.preview_size,self.preview_bgr)
                            if keep_frame else None
                        )
                        copy_end_ns = self.clock.now_ns() if copy_start_ns is not None else None
                        last_total += 1

                    timestamp_ns = self.clock.ticks_to_ns(present_ticks)
                    cadence_gaps = 0
                    duplicate = 0
                    if last_timestamp_ns is not None:
                        delta = timestamp_ns - last_timestamp_ns
                        if delta == 0:
                            duplicate = 1
                        elif not skipped and delta > target_period_ns * 3 // 2:
                            cadence_gaps = max(0, round(delta / target_period_ns) - 1)
                    last_timestamp_ns = timestamp_ns
                    if frame is not None:
                        selected_duplicate = timestamp_ns == last_selected_timestamp_ns
                        last_selected_timestamp_ns = timestamp_ns
                        packet = FramePacket(
                            global_frame_id=last_total - 1,
                            timestamp_ns=timestamp_ns,
                            dxgi_present_ticks=present_ticks,
                            image=frame,
                            is_duplicate=selected_duplicate,
                            capture_copy_start_ns=copy_start_ns,
                            capture_copy_end_ns=copy_end_ns,
                            capture_ready_ns=self.clock.now_ns() if self.measure_latency else None,
                        )
                        self.on_frame(packet)
                    with self._stats_lock:
                        self.stats.frames += 1
                        self.stats.latest_timestamp_ns = timestamp_ns
                        self.stats.source_cadence_gaps += cadence_gaps
                        self.stats.duplicate_timestamps += duplicate
                        wall_span = time.perf_counter() - wall_started
                        if wall_span > 0:
                            self.stats.actual_fps = self.stats.frames / wall_span
        except Exception:
            LOGGER.exception("DXcam ring drain failed")
            with self._stats_lock:
                self.stats.errors += 1
