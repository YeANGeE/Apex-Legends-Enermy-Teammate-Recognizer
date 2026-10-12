"""Bounded functional checks; no HTTP service, training or controller access."""
import io
import json
import subprocess
import tempfile
import threading
import unittest
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw

from c1_webui_video import VideoJobs
from c1_webui import RequestHandler
from fps_recorder.utils import resolve_executable


class FakeEngine:
    checkpoint_path = Path("selected.pt")

    def detect_image(self, image, threshold, render_only=False):
        ImageDraw.Draw(image).rectangle((8, 8, 40, 40), outline="orange", width=3)
        return image, [{"label": "敌人", "score": 0.9, "box": [8, 8, 40, 40]}]


class VideoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.jobs = VideoJobs(FakeEngine(), self.root)

    def tearDown(self):
        self.temp.cleanup()

    def seed(self, identifier="test"):
        folder = self.root / identifier
        folder.mkdir()
        self.jobs.jobs[identifier] = (dict(state="processing", frames=0, threshold=.2,
                                         id=identifier), threading.Event())
        self.jobs.busy = True
        return folder

    def test_vfr_audio_export(self):
        folder = self.seed()
        clip = folder / "input.mp4"
        with av.open(str(clip), "w") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width, stream.height = 96, 64
            stream.pix_fmt = "yuv420p"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
            for pts in [0, 3000, 9000, 12000]:
                frame = av.VideoFrame.from_image(Image.new("RGB", (96, 64), "gray"))
                frame.pts, frame.time_base = pts, stream.time_base
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        command = [resolve_executable("ffmpeg"), "-v", "error", "-y", "-i", str(clip),
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=0.17",
                   "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                   "-f", "matroska", str(folder / "source.video")]
        subprocess.run(command, check=True, capture_output=True)
        with av.open(str(folder / "source.video")) as original:
            original_times = [frame.time for frame in original.decode(video=0)]
        self.jobs._run("test")
        status = self.jobs.status("test")
        self.assertEqual(status["state"], "completed", status.get("error"))
        self.assertFalse(self.jobs.busy)
        self.assertFalse((folder / "source.video").exists())
        with av.open(str(folder / "result.mp4")) as output:
            self.assertEqual(len(output.streams.audio), 1)
            frames = list(output.decode(video=0))
        self.assertEqual(len(frames), 4)
        for frame, timestamp in zip(frames, original_times):
            self.assertAlmostEqual(frame.time, timestamp-original_times[0], places=3)
        rows = [json.loads(line) for line in (folder / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["detections"][0]["label"], "敌人")
        self.assertTrue((folder / "preview.jpg").is_file())

    def test_invalid_video(self):
        folder = self.seed()
        (folder / "source.video").write_bytes(b"not a video")
        self.jobs._run("test")
        self.assertEqual(self.jobs.status("test")["state"], "failed")
        self.assertFalse(self.jobs.busy)
        with self.assertRaises(FileNotFoundError):
            self.jobs.file("test", "result.mp4")

    def test_cancel(self):
        folder = self.seed()
        with av.open(str(folder / "source.video"), "w", format="matroska") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width, stream.height = 96, 64
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_image(Image.new("RGB", (96, 64)))
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        self.jobs.cancel("test")
        self.jobs._run("test")
        self.assertEqual(self.jobs.status("test")["state"], "cancelled")
        self.assertFalse(self.jobs.busy)
        self.assertFalse((folder / "result.mp4").exists())

    def test_playback_ranges(self):
        path = self.root / "result.mp4"
        path.write_bytes(b"0123456789")
        for requested, expected in [("bytes=2-5", b"2345"), ("bytes=-3", b"789"),
                                    ("bytes=8-", b"89"), (None, b"0123456789")]:
            handler = RequestHandler.__new__(RequestHandler)
            handler.headers = {"Range": requested} if requested else {}
            handler.wfile = io.BytesIO()
            statuses, headers = [], {}
            handler.send_response = statuses.append
            handler.send_header = lambda key, value: headers.update({key: value})
            handler.end_headers = lambda: None
            handler.send_video_file(path)
            self.assertEqual(handler.wfile.getvalue(), expected)
            self.assertEqual(statuses, [206 if requested else 200])
            self.assertEqual(headers["Content-Length"], str(len(expected)))

    def test_upload_guards(self):
        with self.assertRaises(ValueError):
            self.jobs.upload(io.BytesIO(), 0, .2)
        self.jobs.busy = True
        with self.assertRaises(ValueError):
            self.jobs.upload(io.BytesIO(b"x"), 1, .2)


if __name__ == "__main__":
    unittest.main()
