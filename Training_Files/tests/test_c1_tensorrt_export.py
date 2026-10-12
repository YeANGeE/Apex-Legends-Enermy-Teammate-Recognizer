import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_c1_tensorrt as exporter


class ExportTests(unittest.TestCase):
    def test_fp32_argument_does_not_change_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'best.pt'
            source.write_bytes(b'original')
            captured = {}
            class FakeModel:
                names = {0: 'enemy', 1: 'teammate'}
                def __init__(self, path): self.path = Path(path)
                def export(self, **kwargs):
                    captured.update(kwargs)
                    engine = self.path.with_suffix('.engine')
                    engine.write_bytes(b'fake')
                    return str(engine)
            with patch.object(exporter, 'WEIGHTS', source):
                engine = exporter.export_candidate(FakeModel, root / 'output', {}, precision=32)
            self.assertEqual(captured['quantize'], 32)
            self.assertEqual(exporter.EXPORT_ARGS['quantize'], 16)
            manifest = json.loads((engine.parent / 'export_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['export_args']['quantize'], 32)

    def test_candidate_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'best.pt'
            source.write_bytes(b'original')
            class FakeModel:
                names = {0: 'enemy', 1: 'teammate'}
                def __init__(self, path): self.path = Path(path)
                def export(self, **kwargs):
                    self_path = self.path.with_suffix('.engine')
                    self_path.write_bytes(b'fake engine')
                    return str(self_path)
            with patch.object(exporter, 'WEIGHTS', source):
                engine = exporter.export_candidate(FakeModel, root / 'output', {})
            manifest = json.loads((engine.parent / 'export_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(source.read_bytes(), b'original')
            self.assertEqual(manifest['status'], 'exported')
            self.assertFalse(manifest['promoted'])
            self.assertEqual(manifest['export_args']['batch'], 1)
            self.assertEqual(manifest['export_args']['quantize'], 16)
            self.assertIsNone(manifest['export_args']['nms'])

    def test_wrong_classes_fail_with_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'best.pt'
            source.write_bytes(b'original')
            class WrongModel:
                names = {0: 'other'}
                def __init__(self, path): pass
            with patch.object(exporter, 'WEIGHTS', source), self.assertRaises(ValueError):
                exporter.export_candidate(WrongModel, root / 'output', {})
            report = next((root / 'output').glob('*/export_manifest.json'))
            manifest = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            self.assertTrue(manifest['original_unchanged'])


if __name__ == '__main__': unittest.main()
