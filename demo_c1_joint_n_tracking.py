"""User-run frozen joint-n baseline tracking, split view, no controller access."""
from pathlib import Path
import sys
import demo_c1_video_tracking as demo
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/"runs/c1_yolo/day3_yolo26n_1280_joint_pretrained_v1/weights/best.pt"
EXPECTED_SHA="12c41173c4b20bc8a62ea92faaeb9295c50d43c87a149ca7c176a085337429a2"


if __name__=="__main__":
    if sha(WEIGHTS)!=EXPECTED_SHA:
        raise ValueError("Frozen C1 checkpoint changed; review a new version explicitly")
    demo.WEIGHTS=WEIGHTS
    demo.SOURCE=Path(r"D:\FPS_AI_Dataset\Apex_YOLO_DAY3")
    print("C1视频跟踪基线：联合26n，第8轮best；左侧原始检测，右侧时序跟踪；不控制游戏。",flush=True)
    print("Day3 E43/47/79/82/150/178/320；部分帧已参与C1训练，作为功能诊断，不宣称独立泛化。",flush=True)
    print("目标规则：最低0.05，高置信度>=0.25，大框>=画面1%；只选已确认当前可见敌人，保持有效ID，失效立即重选。",flush=True)
    demo.main(["--episodes","43","47","79","82","150","178","320","--split-view","--select-target",*sys.argv[1:]])
