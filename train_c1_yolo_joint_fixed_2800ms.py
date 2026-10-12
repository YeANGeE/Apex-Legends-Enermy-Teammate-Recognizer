"""User-run new joint n experiment, original COCO initialization and recipe."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import train_c1_yolo_joint_pretrained as baseline
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'c1_yolo_dataset/day3_joint_fixed_2800ms_v1'
SETTINGS={**baseline.SETTINGS,'data':str(DATA/'data.yaml'),'name':'day3_yolo26n_1280_joint_fixed2800_v1'}
EXPECTED_MANIFEST_SHA='ef499410b7f86db6a451dca64d159034138b55c382ac73b03741fbce947a451c'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if sha(DATA/'source_manifest.json')!=EXPECTED_MANIFEST_SHA:raise ValueError('Prepared manifest changed')
    baseline.shared.DATA=DATA;baseline.shared.SETTINGS=SETTINGS
    baseline.shared.check_configuration(weights=baseline.WEIGHTS,initial_classes='coco',expected_sha=baseline.EXPECTED_SHA,
                                        expected_counts={'train':1685,'val':62})
    if (ROOT/'runs/c1_yolo'/SETTINGS['name']).exists():raise FileExistsError('Preserve existing experiment; choose a new version')
    print('本轮：原始通用YOLO26n预训练 + 全部1685张训练图；原62张验证保留。非当前best继续微调。',flush=True)
    print('增加301张不重复图片，并用最新人工框替换1张重复图；其余参数沿用联合n。',flush=True)
    if args.check_only:print('仅检查，未启动训练。',flush=True);return
    receipt=dict(experiment=SETTINGS['name'],initial_checkpoint=str(baseline.WEIGHTS),
                 initial_checkpoint_sha256=baseline.EXPECTED_SHA,manifest_sha256=EXPECTED_MANIFEST_SHA,
                 settings=SETTINGS,promotion_status='candidate; compare errors and small-target recall on fresh videos')
    model=YOLO(str(baseline.WEIGHTS))
    def save_receipt(trainer):
        (Path(trainer.save_dir)/'data_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    model.add_callback('on_train_start',save_receipt)
    model.train(**SETTINGS)
    print(f'候选权重：{model.trainer.best}\n结果：{model.trainer.save_dir}',flush=True)


if __name__=='__main__':main()
