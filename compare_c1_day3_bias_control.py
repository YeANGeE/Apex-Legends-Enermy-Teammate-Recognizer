"""User-run comparison: original V4, previous Day3 candidate, controlled warm-up candidate."""
from pathlib import Path
import compare_c1_external_screenshots as comparison

ROOT=Path(__file__).resolve().parent


if __name__=="__main__":
    comparison.YOLO_WEIGHTS={
        "yolo_day2_v4":comparison.YOLO_WEIGHTS["yolo_day2_v4"],
        "yolo_day3_targeted":ROOT/"runs/c1_yolo/day3_yolo26n_1280_v4_targeted_v1-2/weights/best.pt",
        "yolo_day3_bias_control":ROOT/"runs/c1_yolo/day3_yolo26n_1280_v4_bias_control_v1/weights/best.pt"}
    comparison.DISPLAY.update({"yolo_day3_targeted":"Day3 上轮候选（第2轮）",
                               "yolo_day3_bias_control":"Day3 偏置预热对照（第3轮）"})
    comparison.main(yolo_only=True)
