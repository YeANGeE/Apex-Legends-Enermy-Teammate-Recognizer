"""Tests for human annotation conversion, without training or dataset writes."""
import copy
import unittest

from prepare_c1_day1_teammates_yolo import parse_annotation


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.task = {"id": 1, "annotations": [{"was_cancelled": False, "result": [
            {"from_name": "frame_quality", "type": "choices", "value": {"choices": ["usable"]}},
            {"type": "rectanglelabels", "original_width": 1920, "original_height": 1080,
             "value": {"rectanglelabels": ["teammate"], "x": 10, "y": 20, "width": 20, "height": 40}}]}]}

    def test_class_and_normalized_coordinates(self):
        quality, lines = parse_annotation(self.task, (1920, 1080))
        self.assertEqual(quality, "usable")
        self.assertEqual(lines, ["1 0.2000000000 0.4000000000 0.2000000000 0.4000000000"])
        self.task["annotations"][0]["result"][1]["value"]["rectanglelabels"] = ["enemy"]
        self.assertTrue(parse_annotation(self.task, (1920, 1080))[1][0].startswith("0 "))

    def test_invalid_annotations_fail(self):
        for mutation in ("nan", "outside", "unknown", "dimensions", "multiple", "missing_quality"):
            with self.subTest(mutation=mutation):
                task = copy.deepcopy(self.task)
                result = task["annotations"][0]["result"]
                if mutation == "nan":
                    result[1]["value"]["x"] = float("nan")
                elif mutation == "outside":
                    result[1]["value"]["width"] = 100
                elif mutation == "unknown":
                    result[1]["value"]["rectanglelabels"] = ["person"]
                elif mutation == "dimensions":
                    result[1]["original_width"] = 1280
                elif mutation == "multiple":
                    task["annotations"].append(copy.deepcopy(task["annotations"][0]))
                else:
                    result.pop(0)
                with self.assertRaises(ValueError):
                    parse_annotation(task, (1920, 1080))

    def test_uncertain_and_empty_frames(self):
        result = self.task["annotations"][0]["result"]
        result.pop()
        self.assertEqual(parse_annotation(self.task, (1920, 1080)), ("usable", []))
        result[0]["value"]["choices"] = ["uncertain"]
        self.assertEqual(parse_annotation(self.task, (1920, 1080)), ("uncertain", []))


if __name__ == "__main__":
    unittest.main()
