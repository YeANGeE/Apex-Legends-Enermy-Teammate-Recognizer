"""Interface smoke test with random architecture, NOT trained model accuracy."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ultralytics import YOLO
from c1_tracking_advanced import AppearanceEncoder


class EncoderTest(unittest.TestCase):
    def test_batch_embeddings_and_small_crop_without_download(self):
        factory=YOLO
        with patch('ultralytics.YOLO',side_effect=lambda _:factory('yolo26n-cls.yaml')):
            encoder=AppearanceEncoder('synthetic-only.pt',device='cpu')
        rgb=np.random.default_rng(2).integers(0,256,(360,640,3),dtype=np.uint8)
        boxes=[[100,100,140,200],[10,10,50,80],[1,1,4,6]]
        features=encoder.extract(rgb,[dict(box=b) for b in boxes])
        self.assertEqual(len(features),3)
        for feature in features[:2]:
            self.assertIsNotNone(feature)
            self.assertAlmostEqual(float(np.linalg.norm(feature)),1.,places=5)
        self.assertIsNone(features[2])


if __name__=='__main__':unittest.main()
