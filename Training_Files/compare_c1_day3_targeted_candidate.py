"""PyCharm entry: original V4 vs Day3 best candidate, same 27 human-labelled images."""
from pathlib import Path
import compare_c1_external_screenshots as comparison

ROOT=Path(__file__).resolve().parent
CANDIDATE=ROOT/"runs/c1_yolo/day3_yolo26n_1280_v4_targeted_v1-2/weights/best.pt"


if __name__=="__main__":
    comparison.YOLO_WEIGHTS={"yolo_day2_v4":comparison.YOLO_WEIGHTS["yolo_day2_v4"],
                             "yolo_day3_targeted":CANDIDATE}
    comparison.DISPLAY["yolo_day3_targeted"]="YOLO Day3 补强候选（第2轮）"
    comparison.main(yolo_only=True)
