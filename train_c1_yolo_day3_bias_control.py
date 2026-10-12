"""User-run single-variable control: cap bias warm-up LR for V4 fine-tuning.

Same V4 initialization/data/seed/optimizer/augmentation as the previous Day3 run.
Only warmup_bias_lr changes; new output name preserves the completed experiment.
"""
import train_c1_yolo_day3_targeted as training


if __name__ == "__main__":
    training.SETTINGS = {**training.SETTINGS, "warmup_bias_lr": 0.0001,
                         "name": "day3_yolo26n_1280_v4_bias_control_v1"}
    print("单变量对照：warmup_bias_lr由默认0.1改为0.0001，其余训练条件保持。", flush=True)
    print("从原V4 best.pt重新初始化；不是在上轮候选上继续训练。", flush=True)
    training.main()
