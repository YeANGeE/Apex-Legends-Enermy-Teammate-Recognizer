"""User-run supplement candidate on reserved DAY4 clips, geometry baseline."""
import json
import sys
from pathlib import Path
import demo_c1_video_tracking as demo
from normalize_c1_day3_export import sha

ROOT = Path(__file__).resolve().parent
SOURCE = Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY4')
EPISODES = [1, 3, 6, 7, 9]
WEIGHTS = ROOT/'runs/c1_yolo/day4_yolo26n_1280_joint_supplement_v1/weights/best.pt'
EXPECTED_SHA = 'a2ceeb4de51d51d9a422014de3a0a657aeba5c7f645e09188a584227721a93c7'
TRAIN = ROOT/'c1_yolo_dataset/day4_joint_supplement_v1/source_manifest.json'
TRAIN_SHA = '214e089aa1241c4da76bfddde1ac1194d3d6ead7bb82059cf5d5d34d7d0bc826'


def check_sources():
    if sha(WEIGHTS) != EXPECTED_SHA or sha(TRAIN) != TRAIN_SHA:
        raise ValueError('Candidate weights or dataset manifest changed')
    receipt = json.loads((WEIGHTS.parent.parent/'data_receipt.json').read_text(encoding='utf-8'))
    if receipt['manifest_sha256'] != TRAIN_SHA:
        raise ValueError('Training receipt differs')
    report = json.loads((ROOT/'diagnostics/c1_day4_unseen_validation_v1/selection_manifest.json').read_text(encoding='utf-8'))
    if report['episodes'] != EPISODES:
        raise ValueError('Reserved episode roster changed')
    train = json.loads(TRAIN.read_text(encoding='utf-8'))
    train_paths = {str(Path(r['episode_path']).resolve()).lower() for r in train if r.get('episode_path')}
    train_videos = {r['source_video_sha256'] for r in train if r.get('source_video_sha256')}
    for audit in report['clips']:
        path = SOURCE/audit['episode']
        if str(path.resolve()).lower() in train_paths or audit['source_video_sha256'] in train_videos:
            raise ValueError('Reserved clip used in training')
        for name, key in [('video.mkv','source_video_sha256'), ('metadata.json','source_metadata_sha256'), ('frames.csv','source_frames_sha256')]:
            if sha(path/name) != audit[key]:
                raise ValueError('Reserved source changed')


if __name__ == '__main__':
    if any(flag in sys.argv for flag in ('--source','--detector','--episodes','--group-duplicates','--tracker')):
        raise ValueError('Keep sources, detector and geometry-only tracker fixed for this comparison')
    if demo.DETECTION_CONFIDENCE != .10:
        raise ValueError('Comparison requires detection threshold 0.10')
    check_sources()
    demo.WEIGHTS = WEIGHTS
    demo.SOURCE = SOURCE
    print('DAY4补样模型（CSV最佳第11轮）：同E1/E3/E6/E7/E9，阈值0.10、面积修正开启、重复框分组关闭。', flush=True)
    print('对照旧模型面积修正版20261010_174726_891310；重点复核E3帧230/280，E7场景，E9帧24/364/413/614/642。', flush=True)
    demo.main(['--episodes', *map(str, EPISODES), '--split-view','--select-target','--stable-geometry', *sys.argv[1:]])
