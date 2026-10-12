"""User-run candidate export, never changes the operational detector."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt'
# Stay on the 10.x FP16 builder path; TRT 11 uses a different ModelOpt workflow.
REQUIREMENTS = ['tensorrt-cu13==10.16.1.11', 'onnx>=1.12.0,<2.0.0', 'onnxslim>=0.1.82']
EXPORT_ARGS = dict(format='engine', imgsz=1280, batch=1, dynamic=False,
                   quantize=16, device=0, workspace=4, simplify=True,
                   nms=None, max_det=100)
# In installed Ultralytics 8.4.174, None selects the one-to-many head with
# external NMS (the operational predictor default). False switches to the
# distinct one-to-one / NMS-free head, so it is NOT an equivalent export.


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_dependencies():
    return [name for name in ('tensorrt', 'onnx', 'onnxslim')
            if importlib.util.find_spec(name) is None]


def export_candidate(model_factory, output_root, environment, precision=16):
    """Export a copied checkpoint in a new folder; retain source and provenance."""
    if precision not in (16, 32):
        raise ValueError('仅支持FP16或FP32候选。')
    export_args = dict(EXPORT_ARGS, quantize=precision)
    original_hash = sha256(WEIGHTS)
    folder = output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    copied = folder / 'detector.pt'
    shutil.copy2(WEIGHTS, copied)
    manifest = dict(stage='Stage 1 — Local Combat Policy', source=str(WEIGHTS),
                    source_sha256=original_hash, export_args=export_args,
                    environment=environment, status='building',
                    promoted=False, training_started=False,
                    note='Candidate only; speed/accuracy not validated. No controller output.')
    report = folder / 'export_manifest.json'
    report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        model = model_factory(str(copied))
        if model.names != {0: 'enemy', 1: 'teammate'}:
            raise ValueError('权重类别不是 enemy / teammate，停止导出。')
        engine = Path(model.export(**export_args)).resolve()
        if engine.parent != folder.resolve() or engine.suffix != '.engine' or not engine.is_file():
            raise RuntimeError('导出未生成预期目录内的 .engine 文件。')
        manifest.update(status='exported', engine=str(engine), engine_sha256=sha256(engine))
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        unchanged = sha256(WEIGHTS) == original_hash
        manifest['original_unchanged'] = unchanged
        report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if not unchanged:
            raise RuntimeError('原权重在导出期间发生变化，请检查后停止后续步骤。')
    return engine


def main():
    parser = argparse.ArgumentParser(description='C1 TensorRT FP16候选导出；不训练、不切换正式模型')
    parser.add_argument('--install', action='store_true', help='仅安装导出依赖，然后退出')
    parser.add_argument('--check-only', action='store_true', help='仅检查文件与依赖，不导出')
    parser.add_argument('--precision', type=int, choices=[16, 32], default=16,
                        help='默认16用于加速候选；32用于排查数值/转换差异')
    args = parser.parse_args()
    if args.install and args.check_only:
        parser.error('install 与 check-only 请分别运行')
    print(f'解释器：{sys.executable}', flush=True)
    if Path(sys.prefix).resolve() != (ROOT / '.venv-yolo').resolve():
        raise RuntimeError('请使用项目 .venv-yolo 解释器，避免安装到其他环境。')
    if args.install:
        print('安装 TensorRT CUDA13 10.16.1.11 / ONNX / ONNXSlim；不升级 PyTorch 或 Ultralytics。', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', *REQUIREMENTS], check=True)
        print('依赖安装完成。清空运行参数后，再运行本脚本导出。')
        return
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    missing = missing_dependencies()
    print(f'原权重：{WEIGHTS}\n缺少依赖：{missing or "无"}', flush=True)
    if args.check_only:
        return
    if missing:
        raise RuntimeError('先在 PyCharm 使用 --install 参数运行本脚本。')
    # Disable implicit package installation inside the exporter.
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import torch
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，停止导出。')
    if not trt.__version__.startswith('10.16.1'):
        raise RuntimeError(f'TensorRT版本为 {trt.__version__}，本步骤要求10.16.1；先运行 --install。')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError('TensorRT Builder 初始化失败，检查驱动/CUDA。')
    del builder
    environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       tensorrt=trt.__version__, ultralytics=ultralytics.__version__,
                       gpu=torch.cuda.get_device_name(0))
    print(f'开始构建：1280×1280、batch=1、FP{args.precision}；保留原模型候选框＋外部NMS分支。可能需要数分钟；这不是训练或推理测速。', flush=True)
    engine = export_candidate(YOLO, ROOT / 'runs/c1_tensorrt', environment, precision=args.precision)
    print(f'候选引擎：{engine}\n原模型未切换。下一步检查检测一致性与检测＋跟踪耗时。', flush=True)


if __name__ == '__main__':
    main()
