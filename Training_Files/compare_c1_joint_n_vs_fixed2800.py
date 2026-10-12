"""User-run old joint n vs fixed2800 candidate on the same 27 diagnostic images."""
from pathlib import Path
import compare_c1_external_screenshots as comparison
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
WEIGHTS={
    'joint_n':ROOT/'runs/c1_yolo/day3_yolo26n_1280_joint_pretrained_v1/weights/best.pt',
    'joint_fixed2800':ROOT/'runs/c1_yolo/day3_yolo26n_1280_joint_fixed2800_v1/weights/best.pt'}
HASHES={'joint_n':'12c41173c4b20bc8a62ea92faaeb9295c50d43c87a149ca7c176a085337429a2',
        'joint_fixed2800':'76cc615ca2c6ced6566fd42cd137ef8744478fb8a9886dd1b9c65459327ace9c'}


if __name__=='__main__':
    for key,path in WEIGHTS.items():
        if sha(path)!=HASHES[key]:raise ValueError(f'Checkpoint changed: {key}')
    comparison.YOLO_WEIGHTS=WEIGHTS
    comparison.DISPLAY.update(joint_n='旧联合n（第8轮）',joint_fixed2800='2.8秒补充联合n（第15轮）')
    print('同27张截图、同检测参数；人工标签存在漏标，未匹配框仍需视觉复核。',flush=True)
    comparison.main(yolo_only=True)
