"""User-run duplicate grouping comparison against the stable-geometry run."""
import runpy
import sys
from pathlib import Path
import demo_c1_video_tracking as demo

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    if demo.DETECTION_CONFIDENCE != .10:
        raise ValueError('Comparison requires detection threshold 0.10')
    if any(flag in sys.argv for flag in ('--detector', '--source', '--tracker')):
        raise ValueError('Keep detector, source and tracker fixed for this comparison')
    print('对比试验：保留stable-geometry，只增加group-duplicates；同模型、阈值0.10及五段视频。', flush=True)
    print('敌我双框分组后保持不确定，不按高置信度强制决定敌我；重点检查换号与人物误合并。', flush=True)
    sys.argv = [str(ROOT/'demo_c1_day4_unseen_tracking.py'), '--stable-geometry',
                '--group-duplicates', *sys.argv[1:]]
    runpy.run_path(str(ROOT/'demo_c1_day4_unseen_tracking.py'), run_name='__main__')
