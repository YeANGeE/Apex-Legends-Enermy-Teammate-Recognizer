"""User-run Day4 full joint training with corrected hard examples."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import train_c1_yolo_joint_pretrained as baseline
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'c1_yolo_dataset/day4_joint_errors_v1'
SETTINGS={**baseline.SETTINGS,'data':str(DATA/'data.yaml'),'name':'day4_yolo26n_1280_joint_errors_v1'}
EXPECTED_MANIFEST_SHA='f4e958ffe913040e236fed4fa58fc2e9468c73e11ebe0f9e38c6990737c19a25'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if sha(DATA/'source_manifest.json')!=EXPECTED_MANIFEST_SHA:raise ValueError('Prepared Day4 manifest changed')
    baseline.shared.DATA=DATA;baseline.shared.SETTINGS=SETTINGS
    baseline.shared.check_configuration(weights=baseline.WEIGHTS,initial_classes='coco',expected_sha=baseline.EXPECTED_SHA,
                                        expected_counts={'train':1731,'val':62})
    if (ROOT/'runs/c1_yolo'/SETTINGS['name']).exists():raise FileExistsError('Preserve existing experiment; use a new version')
    print('本轮从通用预训练YOLO26n开始，联合训练全部1731张图片；不是只训练47张补标图，也不继续微调FPS best。',flush=True)
    print('新增46张不同图片；1张重复队友图采用最新人工框，不重复计入。原验证62张保持。',flush=True)
    print('其他参数沿用联合n；本轮先检查误检/敌我/小目标，不同时改变跟踪算法。',flush=True)
    if args.check_only:print('仅检查，未启动训练。',flush=True);return
    receipt=dict(experiment=SETTINGS['name'],initial_checkpoint=str(baseline.WEIGHTS),
                 initial_checkpoint_sha256=baseline.EXPECTED_SHA,manifest_sha256=EXPECTED_MANIFEST_SHA,
                 settings=SETTINGS,dataset_summary_sha256=sha(DATA/'dataset_summary.json'),
                 promotion_status='candidate; reviewed clips are training exposed, fresh video required')
    model=YOLO(str(baseline.WEIGHTS))
    def save_receipt(trainer):
        (Path(trainer.save_dir)/'data_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    model.add_callback('on_train_start',save_receipt)
    model.train(**SETTINGS)
    print(f'候选权重：{model.trainer.best}\n结果：{model.trainer.save_dir}',flush=True)


if __name__=='__main__':main()
