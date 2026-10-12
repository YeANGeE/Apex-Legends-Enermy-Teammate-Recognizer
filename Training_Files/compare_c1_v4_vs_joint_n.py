"""User-run original V4 vs generic-pretrained joint n, best epoch8/AP95=.33314."""
from pathlib import Path
import json
import compare_c1_external_screenshots as comparison
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"runs/c1_yolo/day3_yolo26n_1280_joint_pretrained_v1"


if __name__=="__main__":
    receipt=json.loads((RUN/"data_receipt.json").read_text(encoding="utf-8"))
    if receipt["initial_checkpoint_sha256"]!=sha(ROOT/"weights/yolo26n.pt"):
        raise ValueError("Joint n initial checkpoint differs")
    if receipt["manifest_sha256"]!=sha(ROOT/"c1_yolo_dataset/day3_v4_targeted_v1/source_manifest.json"):
        raise ValueError("Joint n training data receipt differs")
    comparison.YOLO_WEIGHTS={"yolo_day2_v4":comparison.YOLO_WEIGHTS["yolo_day2_v4"],
                             "joint_n":RUN/"weights/best.pt"}
    comparison.DISPLAY["joint_n"]="26n联合训练（第8轮，旧验证AP95=0.3331）"
    comparison.main(yolo_only=True)
