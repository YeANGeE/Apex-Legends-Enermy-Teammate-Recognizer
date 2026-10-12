"""Tiny synthetic media integration test: no actual model inference."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import demo_c1_video_tracking as demo


class FakeModel:
    def predict(self,frame,**kwargs):
        # FFmpeg red RGB -> explicit BGR at inference boundary.
        assert frame[100,100,2]>200 and frame[100,100,0]<20
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[100.,100.,140.,200.]]),
            data=torch.tensor([[100.,100.,140.,200.,.8,0.]]),
            conf=torch.tensor([.8]),cls=torch.tensor([0.])))]


class MediaTest(unittest.TestCase):
    def test_synthetic_encode_decode_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'episode_000001';source.mkdir();output=root/'outputs';output.mkdir()
            exe=demo.resolve_executable('ffmpeg')
            subprocess.run([exe,'-v','error','-f','lavfi','-i','color=c=red:s=1920x1080:r=60','-frames:v','2',
                            '-c:v','libx264','-threads','2',str(source/'video.mkv')],check=True,timeout=30)
            (source/'metadata.json').write_text(json.dumps(dict(recording_status='complete',encoder_queue_overflow=0,
                frame_count=2,capture=dict(width=1920,height=1080),encoder=dict(output_fps=60))),encoding='utf-8')
            with (source/'frames.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['frame_id','source_present_ns'])
                writer.writeheader();writer.writerows([dict(frame_id=0,source_present_ns=100_000_000),dict(frame_id=1,source_present_ns=116_666_666)])
            class FakeAppearance:
                def extract(self,frame,predictions):return [np.array([1.,0.]) for _ in predictions]
            demo.run_clip(FakeModel(),source,output,.2,appearance_encoder=FakeAppearance(),select_target=True)
            clip=output/source.name
            logs=[json.loads(s) for s in (clip/'tracks.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(logs),2)
            self.assertEqual(logs[1]['source_present_ns'],116_666_666)
            self.assertEqual(logs[1]['tracks'][0]['track_id'],1)
            # Two hits are still tentative; confirmed tracks require three hits.
            self.assertIsNone(logs[1]['target_selection']['target_id'])
            self.assertEqual(logs[1]['target_selection']['reason'],'idle')
            self.assertFalse(logs[1]['target_selection']['controller_output'])
            summary=json.loads((clip/'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['controller_output'])
            self.assertTrue(summary['target_selection_enabled'])
            self.assertEqual(summary['video_encoder'],'h264_nvenc')
            self.assertEqual(summary['tracker_version'],4)
            self.assertIn('appearance',summary['phase_timings_ms'])
            self.assertIn('camera_flow_forward',summary['phase_timings_ms'])
            self.assertEqual(summary['camera_profile']['width'],640)
            self.assertEqual(summary['opencv_threads'],4)
            self.assertEqual(logs[1]['association_diagnostics']['assignments'][0]['track_id'],1)
            result=subprocess.run([demo.resolve_executable('ffprobe'),'-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_read_frames','-of','json',str(clip/'跟随演示.mp4')],capture_output=True,text=True,check=True,timeout=30)
            stream=json.loads(result.stdout)['streams'][0]
            self.assertEqual((stream['width'],stream['height'],int(stream['nb_read_frames'])),(1920,1080,2))

    def test_optional_split_layout(self):
        frame=np.zeros((1080,1920,3),dtype=np.uint8)
        self.assertEqual(demo.render(frame,[],[],'test').size,(1920,1080))
        self.assertEqual(demo.render(frame,[],[],'test',split_view=True).size,(3840,1080))

    def test_single_transfer_detection_unpack(self):
        boxes=SimpleNamespace(data=torch.tensor([[10.,20.,30.,40.,.1,1.]]))
        rows=demo.prediction_rows(SimpleNamespace(boxes=boxes))
        self.assertEqual(rows[0]['box'],[10.,20.,30.,40.])
        self.assertEqual(rows[0]['class'],1)
        self.assertEqual(demo.prediction_rows(SimpleNamespace(boxes=SimpleNamespace(data=torch.zeros((0,6))))),[])


if __name__=='__main__':unittest.main()
