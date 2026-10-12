"""Short CPU-only synthetic benchmark plumbing tests, no real clip runs."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_c1_camera_profiles import compare,run_profile
from fps_recorder.utils import resolve_executable


class BenchmarkTests(unittest.TestCase):
    def test_transform_difference_units(self):
        reference=[dict(status='accepted',matrix=[[1.,0.,0.],[0.,1.,0.]])]
        candidate=[dict(status='accepted',matrix=[[1.,0.,5.],[0.,1.,0.]])]
        result=compare(reference,candidate,1920,1080)
        self.assertEqual(result['status_mismatch_frames'],0)
        self.assertEqual(result['mean_five_screen_points_displacement_px']['median'],5.)

    def test_native_time_cpu_only_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            subprocess.run([resolve_executable('ffmpeg'),'-v','error','-f','lavfi','-i','color=c=red:s=640x360:r=60',
                '-frames:v','40','-c:v','libx264','-threads','2',str(path/'video.mkv')],check=True,timeout=30)
            meta=dict(recording_status='complete',encoder_queue_overflow=0,frame_count=40,
                      capture=dict(width=640,height=360),encoder=dict(output_fps=60))
            (path/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
            times=[(i if i<20 else i-1)*16_000_000 for i in range(40)]
            rows=[dict(frame_id=i,source_present_ns=t,detections=[]) for i,t in enumerate(times)]
            with (path/'frames.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['frame_id','source_present_ns']);writer.writeheader()
                writer.writerows(dict(frame_id=i,source_present_ns=t) for i,t in enumerate(times))
            report,results,size=run_profile(path,rows,'threads')
            self.assertEqual(report['unique_frames'],39)
            self.assertEqual(len(results),39)
            self.assertEqual(report['steady_timings_ms']['total']['count'],9)
            self.assertAlmostEqual(report['tail_start_elapsed_s'],times[-1]/3e9)
            self.assertEqual(size,(640,360))


if __name__=='__main__':unittest.main()
