"""User-run official-checkpoint optimizer/schedule-inspired YOLO26s control.

Not a reproduction of COCO training: retain FPS data, 1280/batch4, weak HUD-safe
augmentation and existing loss weights; use public optimizer/schedule parameters
embedded in the original S checkpoint. Start fresh; do not resume old optimizer.
"""
from pathlib import Path
import torch
import train_c1_yolo_joint_pretrained as training

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/"weights/yolo26s.pt"
KEYS=("optimizer","lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr")


if __name__=="__main__":
    checkpoint=torch.load(WEIGHTS,map_location="cpu",weights_only=False)
    source=checkpoint["train_args"]
    recipe={key:source[key] for key in KEYS}
    if recipe["optimizer"]!="MuSGD":
        raise ValueError("Original S checkpoint recipe differs")
    training.WEIGHTS=WEIGHTS
    training.EXPECTED_SHA="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
    training.SETTINGS={**training.SETTINGS,**recipe,"epochs":60,"patience":0,
                       "name":"day3_yolo26s_1280_checkpoint_recipe_control_v1"}
    del checkpoint
    print("采用原S权重记录的公共优化器/调度参数：",recipe,flush=True)
    print("保留1384张训练、1280输入、batch=4、原损失权重和弱增强；60轮关闭早停。不是完整COCO配方复现。",flush=True)
    print("从通用S重新初始化，非resume；多项参数作为一个配方组合改变，不能当成纯学习率单变量实验。",flush=True)
    training.main()
