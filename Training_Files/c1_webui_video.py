"""Video jobs inside the user-started WebUI; source PTS drives export timing."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

from fps_recorder.utils import resolve_executable

MAX_VIDEO_BYTES = 2 * 1024**3


class Cancelled(Exception):
    pass


class VideoJobs:
    def __init__(self, engine, root: Path):
        self.engine, self.root = engine, root
        self.jobs = {}
        self.lock = threading.Lock()
        self.busy = False

    def upload(self, source, length, threshold):
        if not 0 < length <= MAX_VIDEO_BYTES:
            raise ValueError("请选择非空视频，最大 2 GB。")
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("置信度阈值必须在 0.01 到 0.99 之间。")
        with self.lock:
            if self.busy:
                raise ValueError("已有视频正在处理，请完成或取消后再提交。")
            self.busy = True
        identifier = uuid.uuid4().hex
        folder = self.root / identifier
        try:
            folder.mkdir(parents=True)
            with (folder / "source.video").open("wb") as handle:
                remaining = length
                while remaining:
                    data = source.read(min(1024**2, remaining))
                    if not data:
                        raise ValueError("视频上传中断，请重新上传。")
                    handle.write(data)
                    remaining -= len(data)
            job = dict(id=identifier, state="processing", frames=0, total_frames=0,
                       elapsed_s=0, video_time_s=0, threshold=threshold,
                       model_name=None, saved_path=None, error=None,
                       preview_url=None, result_url=None)
            with self.lock:
                self.jobs[identifier] = (job, threading.Event())
            threading.Thread(target=self._run, args=(identifier,), daemon=True).start()
            return self.status(identifier)
        except Exception:
            (folder / "source.video").unlink(missing_ok=True)
            with self.lock:
                self.busy = False
            raise

    def status(self, identifier):
        with self.lock:
            return dict(self.jobs[identifier][0])

    def update(self, identifier, **values):
        with self.lock:
            self.jobs[identifier][0].update(values)

    def cancel(self, identifier):
        with self.lock:
            self.jobs[identifier][1].set()

    def file(self, identifier, name):
        status = self.status(identifier)
        if name == "result.mp4" and status["state"] != "completed":
            raise FileNotFoundError(name)
        path = self.root / identifier / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def _run(self, identifier):
        job, stop = self.jobs[identifier]
        folder = self.root / identifier
        source = folder / "source.video"
        silent = folder / "silent.mp4"
        result = folder / "result.mp4"
        started = time.monotonic()
        process = None
        try:
            ffmpeg = resolve_executable("ffmpeg")
            with av.open(str(source)) as video:
                if not video.streams.video:
                    raise ValueError("文件中没有可解码的视频轨道。")
                stream = video.streams.video[0]
                self.update(identifier, total_frames=stream.frames)
                origin = None
                previous = None
                count = 0
                duration = 0
                with av.open(str(silent), "w", options={"movflags": "+faststart"}) as output, \
                        (folder / "detections.jsonl").open("w", encoding="utf-8") as log:
                    encoder = output.add_stream("libx264", rate=stream.average_rate or 30)
                    encoder.width = stream.codec_context.width + stream.codec_context.width % 2
                    encoder.height = stream.codec_context.height + stream.codec_context.height % 2
                    encoder.pix_fmt = "yuv420p"
                    encoder.time_base = Fraction(1, 90000)
                    encoder.codec_context.time_base = encoder.time_base
                    encoder.options = {"preset": "veryfast", "crf": "20"}
                    for frame in video.decode(stream):
                        if stop.is_set():
                            raise Cancelled()
                        if frame.pts is None:
                            raise ValueError("视频帧缺少时间戳，无法保证播放时间正确。")
                        timestamp = frame.pts * frame.time_base
                        if previous is not None and timestamp <= previous:
                            raise ValueError("视频时间戳不递增，无法可靠导出。")
                        if origin is None:
                            origin = timestamp
                        annotated, detections = self.engine.detect_image(
                            frame.to_image(), job["threshold"], render_only=True)
                        export_image = annotated
                        if annotated.size != (encoder.width, encoder.height):
                            export_image = Image.new("RGB", (encoder.width, encoder.height))
                            export_image.paste(annotated, (0, 0))
                        rendered = av.VideoFrame.from_image(export_image)
                        rendered.time_base = encoder.time_base
                        rendered.pts = round((timestamp-origin) / encoder.time_base)
                        for packet in encoder.encode(rendered):
                            output.mux(packet)
                        count += 1
                        interval = (frame.duration * frame.time_base if frame.duration else
                                    timestamp-previous if previous is not None else
                                    Fraction(1, 1) / (stream.average_rate or 30))
                        duration = float(timestamp-origin+interval)
                        log.write(json.dumps(dict(frame=count-1, source_pts=frame.pts,
                                   source_time_base=str(frame.time_base),
                                   video_time_s=float(timestamp-origin), detections=detections),
                                   ensure_ascii=False)+"\n")
                        if count == 1 or time.monotonic()-last_preview >= 0.5:
                            # Atomic replacement keeps concurrent preview reads complete.
                            annotated.thumbnail((960, 540))
                            annotated.save(folder / "preview.tmp", format="JPEG", quality=85)
                            (folder / "preview.tmp").replace(folder / "preview.jpg")
                            last_preview = time.monotonic()
                        self.update(identifier, frames=count, video_time_s=round(float(timestamp-origin), 3),
                                    elapsed_s=round(time.monotonic()-started, 1),
                                    preview_url=f"/api/video/{identifier}/preview.jpg")
                        previous = timestamp
                    if count == 0:
                        raise ValueError("没有解码出任何视频帧。")
                    for packet in encoder.encode():
                        output.mux(packet)
            self.update(identifier, state="encoding")
            with (folder / "encode.log").open("wb") as errors:
                # Preserve variable frame timestamps; trim/offset source audio to video origin.
                process = subprocess.Popen([
                    ffmpeg, "-v", "error", "-y", "-copyts", "-i", str(silent),
                    "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy", "-af", f"atrim=start={float(origin)},asetpts=PTS-{float(origin)}/TB",
                    "-c:a", "aac", "-t", str(duration), "-movflags", "+faststart", str(result)],
                    stdout=subprocess.DEVNULL, stderr=errors,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while process.poll() is None:
                    if stop.wait(0.2):
                        process.terminate()
                        raise Cancelled()
                if process.returncode:
                    raise RuntimeError("MP4 封装失败：" + (folder / "encode.log").read_text(errors="replace")[-1500:])
            with av.open(str(result)) as exported:
                verified = 0
                for _ in exported.decode(video=0):
                    if stop.is_set():
                        raise Cancelled()
                    verified += 1
            if verified != count:
                raise RuntimeError(f"导出帧数校验失败：{verified}/{count}")
            self.update(identifier, state="completed", frames=count, total_frames=count,
                        saved_path=str(result), result_url=f"/api/video/{identifier}/result.mp4",
                        elapsed_s=round(time.monotonic()-started, 1))
            (folder / "summary.json").write_text(json.dumps(dict(self.status(identifier),
                checkpoint=str(self.engine.checkpoint_path)), ensure_ascii=False, indent=2), encoding="utf-8")
        except Cancelled:
            self.update(identifier, state="cancelled")
        except Exception as error:
            self.update(identifier, state="failed", error=str(error))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            source.unlink(missing_ok=True)
            silent.unlink(missing_ok=True)
            if self.status(identifier)["state"] != "completed":
                result.unlink(missing_ok=True)
            with self.lock:
                self.busy = False
