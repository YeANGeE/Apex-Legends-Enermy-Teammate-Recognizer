"""User-run controlled geometry experiment on the same five unseen DAY4 clips."""
import runpy
import sys
from pathlib import Path
import demo_c1_video_tracking as demo

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    if demo.DETECTION_CONFIDENCE != .10:
        raise ValueError('This comparison requires the accepted detection threshold 0.10')
    if '--group-duplicates' in sys.argv or '--detector' in sys.argv or '--source' in sys.argv:
        raise ValueError('Keep detector, source and duplicate handling fixed for this geometry comparison')
    print('对比试验：仅启用stable-geometry；同五段DAY4视频、同模型、阈值0.10。', flush=True)
    print('检查同人换号是否减少；敌我冲突仍须重新确认，不沿用历史敌人身份。', flush=True)
    sys.argv = [str(ROOT/'demo_c1_day4_unseen_tracking.py'), '--stable-geometry', *sys.argv[1:]]
    runpy.run_path(str(ROOT/'demo_c1_day4_unseen_tracking.py'), run_name='__main__')
