"""User-run joint training from generic YOLO26n pretrained weights; no prior FPS model.

Practical strategy comparison, not a single-variable initialization ablation:
uses a larger learning rate and budget than the V4 fine-tune to learn FPS classes.
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import train_c1_yolo_day3_targeted as shared
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/"weights/yolo26n.pt"
EXPECTED_SHA="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
SETTINGS={**shared.SETTINGS,"lr0":.001,"warmup_bias_lr":.001,"epochs":60,"patience":20,
          "name":"day3_yolo26n_1280_joint_pretrained_v1"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check-only",action="store_true");args=parser.parse_args()
    shared.SETTINGS=SETTINGS
    shared.check_configuration(weights=WEIGHTS,initial_classes="coco",expected_sha=EXPECTED_SHA)
    print(f"起点为通用80类预训练{WEIGHTS.stem}；训练时适配enemy/teammate两类，不加载Day1/Day2/Day3 FPS权重。",flush=True)
    print("相对V4微调属于训练策略对照；各实验条件以上述配置和保存回执为准。",flush=True)
    if args.check_only:
        print("仅检查，未训练。",flush=True);return
    manifest_sha=sha(shared.DATA/"source_manifest.json")
    model=YOLO(str(WEIGHTS))
    receipt={"experiment":WEIGHTS.stem+"_joint_from_generic_pretrained","initial_checkpoint":str(WEIGHTS),
             "initial_checkpoint_sha256":EXPECTED_SHA,"manifest_sha256":manifest_sha,"settings":SETTINGS,
             "comparison_scope":"Practical training strategies, different LR and budget; not pure initialization ablation",
             "promotion_status":"unpromoted; requires independent new match"}
    def save_receipt(trainer):
        (Path(trainer.save_dir)/"data_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    model.add_callback("on_train_start",save_receipt)
    model.train(**SETTINGS)
    print(f"候选权重：{model.trainer.best}\n结果：{model.trainer.save_dir}",flush=True)


if __name__=="__main__":
    main()
