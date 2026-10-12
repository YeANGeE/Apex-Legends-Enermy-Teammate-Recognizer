import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from c1_gallery import Gallery


class GalleryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gallery = Gallery(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_roster_and_thumbnail(self):
        data = self.gallery.listing()
        self.assertEqual(data["total"], 779)
        self.assertEqual([i["id"] for i in data["items"]], list(range(1, 780)))
        self.assertEqual(sum(i["split"] == "val" for i in data["items"]), 62)
        with Image.open(io.BytesIO(self.gallery.image(1, True))) as image:
            self.assertEqual(image.size, (640, 360))
        with self.assertRaises(ValueError):
            self.gallery.image(780, False)

    def test_save_resume_and_validation(self):
        self.gallery.save([779, 1, 1], self.gallery.roster_hash)
        loaded = Gallery(Path(self.temp.name))
        self.assertEqual(loaded.state()["selected_ids"], [1, 779])
        self.assertFalse(loaded.state()["completed"])
        for ids in ([0], [780], [True], ["2"], None):
            with self.assertRaises(ValueError):
                self.gallery.save(ids, self.gallery.roster_hash)
        with self.assertRaises(ValueError):
            self.gallery.save([1], "wrong roster")
        with self.assertRaises(ValueError):
            self.gallery.finalize([1], self.gallery.roster_hash, False)

    def test_finalize_subset_and_no_false_completion(self):
        def mock_create(command, **kwargs):
            target = Path(command[command.index("--output")+1])
            tasks = json.loads(Path(command[command.index("--tasks-json")+1]).read_text())
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0]["meta"]["image_name"], self.gallery.tasks[0]["meta"]["image_name"])
            self.assertIn("predictions", tasks[0])
            (target / "project_info.json").write_text(json.dumps({"url": "http://127.0.0.1:8080/projects/999/data", "title": "test"}))
            return subprocess.CompletedProcess(command, 0, b"ok", b"")
        with patch("c1_gallery.subprocess.run", side_effect=mock_create) as mocked:
            result = self.gallery.finalize([1, 779], self.gallery.roster_hash, True)
            self.assertEqual(result["selected_count"], 2)
            self.assertEqual(result["unselected_count"], 777)
            decisions = json.loads((Path(result["snapshot"]) / "screening_decisions.json").read_text())
            self.assertEqual(decisions[1]["status"], "user_screened_no_teammate")
            self.assertTrue(self.gallery.state()["completed"])
            self.gallery.finalize([1, 779], self.gallery.roster_hash, True)
            self.assertEqual(mocked.call_count, 1)
        self.gallery.save([1], self.gallery.roster_hash)
        self.assertFalse(self.gallery.state()["completed"])


if __name__ == "__main__":
    unittest.main()
