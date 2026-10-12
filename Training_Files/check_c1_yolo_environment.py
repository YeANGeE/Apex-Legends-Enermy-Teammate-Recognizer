"""Check the C1 YOLO environment and reviewed Day1 data without training."""

from pathlib import Path
import sys

import torch
import torchvision
import ultralytics
import yaml


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = root / "c1_yolo_dataset" / "day1_v1_verified" / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("Select the .venv-yolo interpreter with CUDA-enabled PyTorch.")
    print("GPU:", torch.cuda.get_device_name(0))
    # Exercise a CUDA convolution, not just driver/device enumeration.
    conv = torch.nn.Conv2d(3, 8, kernel_size=3).cuda()
    with torch.inference_mode():
        result = conv(torch.zeros(1, 3, 32, 32, device="cuda"))
    torch.cuda.synchronize()
    print("CUDA convolution:", tuple(result.shape), "OK")
    for split, expected in (("train", 478), ("val", 62)):
        image_dir = Path(config["path"]) / config[split]
        label_dir = Path(config["path"]) / "labels" / split
        images = {p.stem for p in image_dir.glob("*.jpg")}
        labels = {p.stem for p in label_dir.glob("*.txt")}
        if len(images) != expected or images != labels:
            raise RuntimeError(f"Invalid {split} image/label counts or pairing.")
        print(f"{split}: {len(images)} images and matching labels OK")
    if config["names"] != {0: "enemy"}:
        raise RuntimeError("Expected the single enemy class.")
    print("Environment and dataset checks passed. No training was started.")


if __name__ == "__main__":
    main()
