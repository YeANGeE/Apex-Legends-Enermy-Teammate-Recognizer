"""GPU smoke/integration test for the local YOLO WebUI; no training."""
import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request

from PIL import Image

from c1_webui import AppServer, DetectionEngine, DEFAULT_THRESHOLD, ROOT


class YoloWebUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DetectionEngine()
        cls.server = AppServer(("127.0.0.1", 0), cls.engine)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.image = next((ROOT / "c1_yolo_dataset/day1_v2_expanded/images/val").glob("*.jpg"))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_status_and_frontend(self):
        with urllib.request.urlopen(self.url + "/api/status") as response:
            status = json.load(response)
        self.assertEqual(status["backend"], "yolo")
        self.assertEqual(status["checkpoint_epoch"], self.engine.epoch)
        self.assertEqual(status["webui_version"], 2)
        self.assertEqual(status["default_threshold"], 0.20)
        with urllib.request.urlopen(self.url) as response:
            html = response.read().decode()
        self.assertIn("YOLO26n", html)
        self.assertIn('value="0.20"', html)
        self.assertNotIn("RCNN", html)

    def test_detection_api(self):
        payload = {"image": "data:image/jpeg;base64," + base64.b64encode(self.image.read_bytes()).decode(),
                   "filename": "webui_smoke_test.jpg"}
        request = urllib.request.Request(self.url + "/api/detect", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        self.assertEqual(result["threshold"], DEFAULT_THRESHOLD)
        self.assertEqual(result["checkpoint_epoch"], self.engine.epoch)
        self.assertGreater(result["inference_ms"], 0)
        self.assertGreaterEqual(result["prediction_ms"], result["inference_ms"])
        self.assertEqual(result["detection_count"], len(result["detections"]))
        image_bytes = base64.b64decode(result["image_data_url"].split(",")[1])
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            self.assertEqual(image.size, (result["image_width"], result["image_height"]))
        for detection in result["detections"]:
            self.assertIn(detection["label"], {"敌人", "队友"})
            self.assertGreaterEqual(detection["score"], DEFAULT_THRESHOLD)
            x1, y1, x2, y2 = detection["box"]
            self.assertTrue(0 <= x1 <= x2 <= result["image_width"])
            self.assertTrue(0 <= y1 <= y2 <= result["image_height"])

    def test_reject_invalid_image(self):
        request = urllib.request.Request(self.url + "/api/detect", data=b'{"image":"bad"}',
                                         headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertEqual(context.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
