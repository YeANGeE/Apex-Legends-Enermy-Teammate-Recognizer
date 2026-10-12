import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import benchmark_c1_tensorrt as benchmark
from benchmark_c1_tensorrt import compare_detections, distribution


def detection(box, cls=0, score=.5):
    return dict(box=box, **{'class': cls}, score=score)


class ComparisonTests(unittest.TestCase):
    def test_reordered_boxes_match(self):
        boxes = [detection([0, 0, 20, 40]), detection([100, 100, 120, 140], 1)]
        result = compare_detections(boxes, boxes[::-1])
        self.assertEqual(result['same_class_matches'], 2)
        self.assertEqual(result['pt_unmatched'], 0)
        self.assertEqual(result['matched_ious'], [1, 1])

    def test_duplicates_are_not_double_matched(self):
        box = detection([0, 0, 20, 40])
        result = compare_detections([box], [box, box])
        self.assertEqual(result['same_class_matches'], 1)
        self.assertEqual(result['engine_unmatched'], 1)

    def test_class_disagreement(self):
        result = compare_detections([detection([0, 0, 20, 40])], [detection([0, 0, 20, 40], 1)])
        self.assertEqual(result['same_class_matches'], 0)
        self.assertEqual(result['spatial_class_disagreements'], 1)

    def test_empty_and_quantiles(self):
        self.assertEqual(compare_detections([], [])['same_class_matches'], 0)
        self.assertEqual(compare_detections([], [detection([0, 0, 20, 40])])['engine_unmatched'], 1)
        self.assertIsNone(distribution([])['median'])
        self.assertEqual(distribution([1, 2, 3])['median'], 2)

    def test_headless_loop_reuses_timestamp_and_excludes_initial_frames(self):
        class Pipe:
            remaining = 33
            def read(self, size):
                if not self.remaining: return b''
                self.remaining -= 1
                return bytes(size)
            def close(self): pass
        class Decoder:
            stdout = Pipe()
            def wait(self, **kwargs): return 0
            def poll(self): return 0
        class Model:
            names = {0: 'enemy', 1: 'teammate'}
            calls = 0
            class predictor:
                class model:
                    end2end = False
            def predict(self, frame, **kwargs):
                self.calls += 1
                return [object()]
        class Appearance:
            def extract(self, frame, detections): return []
        class Tracker:
            timestamp = None
            events = ['camera_accepted']
            diagnostics = {'tracking_timings_ms': {'camera': .1}}
            def __init__(self, **kwargs): pass
            def update_frame(self, timestamp, detections, frame, features):
                self.timestamp = timestamp
                return []
        times = list(range(1, 33)) + [32]
        frames = [dict(frame_id=i, source_present_ns=t) for i, t in enumerate(times)]
        model = Model()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(benchmark, 'load_clip', return_value=({'capture': {'width': 1920, 'height': 1080}}, frames)), \
             patch.object(benchmark, 'YOLO', return_value=model), \
             patch.object(benchmark, 'AdvancedScreenTracker', Tracker), \
             patch.object(benchmark, 'prediction_rows', return_value=[]), \
             patch.object(benchmark, 'resolve_executable', return_value='ffmpeg'), \
             patch.object(benchmark.subprocess, 'Popen', return_value=Decoder()), \
             patch.object(benchmark.torch.cuda, 'synchronize'), \
             patch.object(benchmark.torch.cuda, 'empty_cache'):
            summary, records = benchmark.run_backend('fake', Path('fake.pt'), Path('episode'),
                Path(folder), Appearance(), benchmark.CAMERA_PROFILES['threads'], 10)
            self.assertEqual(model.calls, 42)  # 10 warmup + 32 distinct native times
            self.assertEqual(len(records), 32)
            self.assertEqual(summary['counts']['reused_timestamp_frames'], 1)
            self.assertEqual(summary['timings_ms']['visual_pipeline']['count'], 2)
            self.assertIsNone(summary['predict_args']['nms'])
            self.assertFalse(summary['runtime_end2end'])
            self.assertEqual(list(Path(folder).iterdir())[0].suffix, '.jsonl')


if __name__ == '__main__': unittest.main()
