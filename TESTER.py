import torch
import torchvision
from torchvision.ops import nms

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

boxes = torch.tensor(
    [
        [0.0, 0.0, 100.0, 100.0],
        [10.0, 10.0, 90.0, 90.0],
        [200.0, 200.0, 300.0, 300.0],
    ],
    device="cuda",
)

scores = torch.tensor(
    [0.95, 0.80, 0.70],
    device="cuda",
)

keep = nms(boxes, scores, iou_threshold=0.5)
torch.cuda.synchronize()

print("NMS result:", keep)
print("NMS device:", keep.device)

assert keep.tolist() == [0, 2]
assert keep.device.type == "cuda"

print("CUDA torchvision detection ops: PASS")