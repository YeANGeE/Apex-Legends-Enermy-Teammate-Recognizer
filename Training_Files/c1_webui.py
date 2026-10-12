"""Local WebUI for testing the current C1 Enemy Detection checkpoint.

Run this file in PyCharm with the .venv-yolo Python interpreter. The
server binds to localhost, keeps uploads in memory, and never modifies datasets.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from c1_gallery import Gallery
from c1_webui_video import VideoJobs, MAX_VIDEO_BYTES


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "c1_webui_static"
CHECKPOINT = ROOT / "runs" / "c1_yolo" / "day4_yolo26n_1280_joint_supplement_v1" / "weights" / "best.pt"
OUTPUT_ROOT = ROOT / "c1_webui_outputs"
DEFAULT_THRESHOLD = 0.20
IMAGE_SIZE = 1280
MODEL_NAME = "YOLO26n · Day4 联合数据补充训练版"
WEBUI_VERSION = 2
CLASS_LABELS = {"enemy": "敌人", "teammate": "队友"}
CLASS_COLORS = {"enemy": (255, 184, 76), "teammate": (76, 210, 255)}
MAX_REQUEST_BYTES = 30 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_detection(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    box: list[float],
    score: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    class_name: str = "enemy",
) -> None:
    x1, y1, x2, y2 = box
    color = CLASS_COLORS[class_name]
    shadow = (18, 24, 32)
    line_width = max(3, round(min(image_size) / 360))
    corner = max(12, round(min(x2 - x1, y2 - y1) * 0.22))

    # A quiet full outline plus stronger targeting corners keeps small enemies
    # legible without covering the underlying frame.
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    segments = (
        ((x1, y1), (x1 + corner, y1)), ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)), ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)), ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)), ((x2, y2), (x2, y2 - corner)),
    )
    for start, end in segments:
        draw.line((start, end), fill=color, width=line_width * 2)

    label = f"{CLASS_LABELS[class_name]}  {score:.2f}"
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_y = max(0, y1 - text_height - 14)
    draw.rounded_rectangle(
        (x1, label_y, x1 + text_width + 18, label_y + text_height + 10),
        radius=5,
        fill=shadow,
        outline=color,
        width=2,
    )
    draw.text((x1 + 9, label_y + 4), label, fill=color, font=font)


class DetectionEngine:
    def __init__(self, checkpoint_path: Path = CHECKPOINT) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用。请使用 .venv-yolo 的 Python 解释器。"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到模型 checkpoint：{checkpoint_path}")

        self.device = torch.device("cuda:0")
        self.model = YOLO(str(checkpoint_path))
        if self.model.names != {0: "enemy", 1: "teammate"}:
            raise ValueError("当前 WebUI 需要 enemy、teammate 双类 YOLO 模型。")
        # Final YOLO checkpoints strip the epoch. Recover provenance from run CSV.
        self.epoch = None
        results_csv = checkpoint_path.parent.parent / "results.csv"
        if checkpoint_path.name == "best.pt" and results_csv.is_file():
            with results_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
            best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
            self.epoch = int(best["epoch"])
        self.checkpoint_path = checkpoint_path
        self.gpu_name = torch.cuda.get_device_name(0)
        self.lock = threading.Lock()
        # Prepare the FP16 backend and warm the exact batch-1 prediction path.
        with torch.inference_mode():
            for _ in range(5):
                self.predict(Image.new("RGB", (1920, 1080)), DEFAULT_THRESHOLD)
        torch.cuda.synchronize()

    def predict(self, image: Image.Image, threshold: float):
        return self.model.predict(source=image, imgsz=IMAGE_SIZE, rect=False,
                                  device=0, quantize=16, conf=threshold, iou=0.5,
                                  max_det=100, verbose=False, nms=None)[0]

    def detect(
        self,
        image_bytes: bytes,
        threshold: float,
        source_name: str = "uploaded_image",
        save_output: bool = True,
    ) -> dict:
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("置信度阈值必须在 0.01 到 0.99 之间。")
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                if source.format not in ALLOWED_IMAGE_FORMATS:
                    raise ValueError(
                        f"不支持的图片格式：{source.format}。请使用 JPG、PNG、WEBP 或 BMP。"
                    )
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(f"无法读取图片：{error}") from error

        return self.detect_image(image, threshold, source_name, save_output)

    def detect_image(self, image, threshold, source_name="video_frame", save_output=False,
                     render_only=False):
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("置信度阈值必须在 0.01 到 0.99 之间。")

        with self.lock, torch.inference_mode():
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = self.predict(image, threshold)
            torch.cuda.synchronize()
            prediction_ms = (time.perf_counter() - started) * 1000
            inference_ms = float(output.speed["inference"])
            boxes = output.boxes.xyxy.detach().cpu()
            scores = output.boxes.conf.detach().cpu()
            classes = output.boxes.cls.detach().cpu()
        detections = [
            {
                "class_id": int(class_id),
                "class_name": self.model.names[int(class_id)],
                "label": CLASS_LABELS[self.model.names[int(class_id)]],
                "score": float(score),
                "box": [round(float(value), 2) for value in box],
            }
            for box, score, class_id in zip(boxes, scores, classes, strict=True)
        ]

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = chinese_font(max(18, round(min(image.size) / 48)))
        for detection in detections:
            draw_detection(draw, image.size, detection["box"], detection["score"], font,
                           detection["class_name"])

        if render_only:
            return annotated, detections

        output_buffer = io.BytesIO()
        annotated.save(output_buffer, format="JPEG", quality=93, optimize=True)
        output_bytes = output_buffer.getvalue()
        output_path: Path | None = None
        if save_output:
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            safe_stem = re.sub(
                r"[^0-9A-Za-z_-]+", "_", Path(source_name).stem
            ).strip("_")
            safe_stem = safe_stem[:80] or "uploaded_image"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = OUTPUT_ROOT / f"{safe_stem}_detected_{timestamp}.jpg"
            output_path.write_bytes(output_bytes)
        encoded = base64.b64encode(output_bytes).decode("ascii")
        return {
            "image_data_url": f"data:image/jpeg;base64,{encoded}",
            "detection_count": len(detections),
            "detections": detections,
            "threshold": threshold,
            "inference_ms": round(inference_ms, 1),
            "prediction_ms": round(prediction_ms, 1),
            "timing_note": "模型前向不含预处理和后处理；检测流程含预处理、传输和后处理，不含上传、解码及绘图。",
            "model_name": MODEL_NAME,
            "image_width": image.width,
            "image_height": image.height,
            "checkpoint_epoch": self.epoch,
            "gpu": self.gpu_name,
            "saved_path": str(output_path) if output_path else None,
        }


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: DetectionEngine,
                 enable_gallery: bool = True) -> None:
        super().__init__(address, RequestHandler)
        self.engine = engine
        self.gallery = Gallery() if enable_gallery else None
        self.videos = VideoJobs(engine, OUTPUT_ROOT / "videos")


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer

    static_types = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def log_message(self, message: str, *args: object) -> None:
        print(f"[WebUI] {self.address_string()} — {message % args}")

    def send_bytes(
        self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if (route == "/api/gallery" or route.startswith("/api/gallery/")
                or route == "/gallery.html") and self.server.gallery is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Sample screening is disabled")
            return
        if route.startswith("/api/video/"):
            parts = route.strip("/").split("/")
            try:
                if len(parts) == 3:
                    self.send_json(self.server.videos.status(parts[2]))
                elif len(parts) == 4 and parts[3] in {"preview.jpg", "result.mp4"}:
                    self.send_video_file(self.server.videos.file(parts[2], parts[3]))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (KeyError, FileNotFoundError):
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if route == "/api/gallery":
            self.send_json(self.server.gallery.listing())
            return
        if route.startswith("/api/gallery/image/"):
            try:
                identifier = int(route.rsplit("/", 1)[1])
                data = self.server.gallery.image(identifier, "thumbnail=1" in self.path)
                self.send_bytes(data, "image/jpeg")
            except (ValueError, OSError):
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if route == "/api/status":
            self.send_json(
                {
                    "ready": True,
                    "gpu": self.server.engine.gpu_name,
                    "checkpoint_epoch": self.server.engine.epoch,
                    "default_threshold": DEFAULT_THRESHOLD,
                    "label": "敌人 / 队友",
                    "classes": CLASS_LABELS,
                    "backend": "yolo",
                    "model_name": MODEL_NAME,
                    "checkpoint": str(self.server.engine.checkpoint_path),
                    "gallery_version": 1,
                    "gallery_enabled": self.server.gallery is not None,
                    "webui_version": WEBUI_VERSION,
                    "video_max_bytes": MAX_VIDEO_BYTES,
                }
            )
            return
        static_name = "index.html" if route == "/" else route.lstrip("/")
        static_path = (STATIC_ROOT / static_name).resolve()
        try:
            static_path.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not static_path.is_file() or static_path.suffix not in self.static_types:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_bytes(static_path.read_bytes(), self.static_types[static_path.suffix])

    def send_video_file(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            a, b = match.groups()
            start, end = (int(a), min(int(b) if b else end, end)) if a else (max(0, size-int(b)), end)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", "video/mp4" if path.suffix == ".mp4" else "image/jpeg")
        self.send_header("Content-Length", str(end-start+1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end-start+1
            while remaining:
                data = handle.read(min(1024*1024, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

    def do_POST(self) -> None:
        if self.path.startswith("/api/gallery/") and self.server.gallery is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Sample screening is disabled")
            return
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.send_json({"error": "只允许从本地 WebUI 提交。"}, HTTPStatus.FORBIDDEN)
            return
        if self.path == "/api/video" or re.fullmatch(r"/api/video/[0-9a-f]{32}/cancel", self.path):
            try:
                if self.path.endswith("/cancel"):
                    self.server.videos.cancel(self.path.split("/")[3])
                    self.send_json({"cancelled": True})
                else:
                    length = int(self.headers.get("Content-Length", "0"))
                    threshold = float(self.headers.get("X-Threshold", DEFAULT_THRESHOLD))
                    job = self.server.videos.upload(self.rfile, length, threshold)
                    self.send_json(job, HTTPStatus.ACCEPTED)
            except (ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        if self.path not in {"/api/detect", "/api/gallery/selection", "/api/gallery/finalize"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求为空或图片超过 30 MB。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求必须为 JSON 对象。")
            if self.path.startswith("/api/gallery/"):
                ids = payload.get("selected_ids")
                roster_hash = payload.get("roster_hash")
                if self.path.endswith("/finalize"):
                    result = self.server.gallery.finalize(ids, roster_hash, payload.get("reviewed_all"))
                else:
                    result = self.server.gallery.save(ids, roster_hash)
                self.send_json(result)
                return
            image_data_url = payload.get("image")
            if not isinstance(image_data_url, str) or "," not in image_data_url:
                raise ValueError("没有收到有效图片。")
            encoded = image_data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded, validate=True)
            threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
            source_name = str(payload.get("filename", "uploaded_image"))
            self.send_json(
                self.server.engine.detect(image_bytes, threshold, source_name)
            )
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            print(f"[WebUI] 推理失败：{error!r}")
            self.send_json(
                {"error": f"模型推理失败：{error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C1 敌人检测本地 WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-gallery", action="store_true",
                        help="关闭依赖本地标注数据的样本筛选，仅提供图片和视频检测")
    parser.add_argument("--self-test-image", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"正在加载 {MODEL_NAME} 最佳模型……", flush=True)
    engine = DetectionEngine()
    print(
        f"模型已就绪：epoch {engine.epoch} | {engine.gpu_name} | 默认阈值 {DEFAULT_THRESHOLD:.2f}",
        flush=True,
    )
    if args.self_test_image:
        result = engine.detect(
            args.self_test_image.read_bytes(), DEFAULT_THRESHOLD, args.self_test_image.name
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "image_data_url"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    server = AppServer((args.host, args.port), engine, enable_gallery=not args.no_gallery)
    url = f"http://{args.host}:{args.port}"
    print(f"WebUI：{url}")
    print(f"识别结果自动保存到：{OUTPUT_ROOT}")
    print("原始上传图片不会被修改。按 PyCharm 红色 Stop 按钮停止服务。")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
