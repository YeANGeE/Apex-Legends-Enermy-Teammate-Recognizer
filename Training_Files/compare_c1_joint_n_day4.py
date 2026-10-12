"""User-run three-generation detector comparison on unchanged 27 screenshots."""
from pathlib import Path
import compare_c1_joint_n_vs_fixed2800 as previous
import compare_c1_external_screenshots as comparison
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
WEIGHTS={**previous.WEIGHTS,'joint_day4':ROOT/'runs/c1_yolo/day4_yolo26n_1280_joint_errors_v1/weights/best.pt'}
HASHES={**previous.HASHES,'joint_day4':'034e5d999ec96ea91778854d81eec881c3da4d89f397a165d872f6e68dc19945'}


if __name__=='__main__':
    for key,path in WEIGHTS.items():
        if sha(path)!=HASHES[key]:raise ValueError(f'Checkpoint changed: {key}')
    comparison.YOLO_WEIGHTS=WEIGHTS
    comparison.DISPLAY.update(joint_n='原联合n（第8轮）',joint_fixed2800='2.8秒补充n（第15轮）',joint_day4='DAY4错误补标n（第33轮）')
    print('同27张诊断图对比三代n；标签存在漏标，未匹配检测仍需人工复核。',flush=True)
    comparison.main(yolo_only=True)
