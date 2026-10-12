"""User-run accuracy and full-resolution screenshots for the completed model registry."""
import compare_c1_external_screenshots as comparison
from benchmark_c1_all_models import CONFIG,load_models


if __name__=="__main__":
    models=load_models(CONFIG)
    comparison.YOLO_WEIGHTS={r["key"]:r["weights"] for r in models}
    comparison.DISPLAY.update({"v4":"原V4","day3_finetune":"Day3微调","day3_bias_control":"偏置预热对照",
                               "joint_n":"26n联合训练","joint_s":"26s联合训练","joint_m":"26m联合训练",
                               "joint_n_full60":"26n完整60轮对照"})
    comparison.main(yolo_only=True)
