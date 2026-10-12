"""User-run Day4 n on new DAY4 recordings, separate from training regression."""
import json
import sys
from pathlib import Path
import demo_c1_video_tracking as demo
from demo_c1_day4_tracking import WEIGHTS,EXPECTED_SHA
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY4')
EPISODES=[1,3,6,7,9]
MANIFEST=ROOT/'diagnostics/c1_day4_unseen_validation_v1/selection_manifest.json'


if __name__=='__main__':
    if sha(WEIGHTS)!=EXPECTED_SHA:raise ValueError('Day4 model changed')
    report=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if report['episodes']!=EPISODES or report['training_source_match']:raise ValueError('Unseen clip selection differs')
    if sha(ROOT/'c1_yolo_dataset/day4_joint_errors_v1/source_manifest.json')!=report['train_manifest_sha256']:
        raise ValueError('Training dataset changed; re-audit clip exposure')
    for audit in report['clips']:
        path=SOURCE/audit['episode']
        for name,key in [('metadata.json','source_metadata_sha256'),('frames.csv','source_frames_sha256'),('video.mkv','source_video_sha256')]:
            if sha(path/name)!=audit[key]:raise ValueError('Audited source changed')
    demo.WEIGHTS=WEIGHTS;demo.SOURCE=SOURCE
    print('DAY4新录像 E1/E3/E6/E7/E9：DAY4模型第33轮best，检测/跟踪参数保持。',flush=True)
    print('未加入训练；左原始检测、右跟踪及目标选择，原尺寸3840×1080，不控制游戏。',flush=True)
    print('完整性提示见diagnostics/c1_day4_unseen_validation_v1/selection_manifest.json。人工看误检、漏检、敌我与ID连续性。',flush=True)
    demo.main(['--episodes',*[str(n) for n in EPISODES],'--split-view','--select-target',*sys.argv[1:]])
