"""User-run resume from an immutable live snapshot; stop original S in PyCharm first."""
import argparse
import json
import math
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check-only",action="store_true");args=parser.parse_args()
    info=json.loads((ROOT/"c1_s_resume_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint=Path(info["checkpoint"])
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if state.get("optimizer") is None or int(state.get("epoch",-1))+1!=info["epoch_completed"]:
        raise ValueError("Snapshot lacks resumable optimizer/epoch")
    if state["train_args"]["epochs"]!=60:
        raise ValueError("Snapshot has a different learning-rate schedule")
    source=Path(info["source_run"])
    old_receipt=json.loads((source/"data_receipt.json").read_text(encoding="utf-8"))
    if old_receipt["manifest_sha256"]!=sha(ROOT/"c1_yolo_dataset/day3_v4_targeted_v1/source_manifest.json"):
        raise ValueError("Training data changed")
    print(f"快照检查通过：已完成第{info['epoch_completed']}轮，保留优化器/EMA；将从下一轮续训到总计60轮，关闭早停。",flush=True)
    if args.check_only:return
    output=ROOT/"runs/c1_yolo/day3_yolo26s_1280_joint_pretrained_resume_full60_v1"
    if output.exists():
        raise FileExistsError("Preserve existing resumed experiment; inspect before rerun")
    output.mkdir(parents=True)
    (output/"weights").mkdir()
    shutil.copy2(checkpoint.parent/"best_snapshot.pt",output/"weights/best.pt")
    shutil.copy2(checkpoint.parent/"results_snapshot.csv",output/"results.csv")
    receipt={**old_receipt,"resume_checkpoint":str(checkpoint),"resume_checkpoint_sha256":sha(checkpoint),
             "resumed_after_epoch":info["epoch_completed"],"settings":{**old_receipt["settings"],"patience":0},
             "note":"Interrupted/resumed experiment, not guaranteed identical to an uninterrupted full60 run."}
    def on_start(trainer):
        if not math.isinf(trainer.stopper.patience):
            raise ValueError("Resume did not disable early stopping")
        (output/"data_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    del state
    model=YOLO(str(checkpoint))
    model.add_callback("on_train_start",on_start)
    model.train(resume=True,patience=0,save_dir=str(output))
    print(f"候选权重：{model.trainer.best}\n结果：{model.trainer.save_dir}",flush=True)


if __name__=="__main__":main()
