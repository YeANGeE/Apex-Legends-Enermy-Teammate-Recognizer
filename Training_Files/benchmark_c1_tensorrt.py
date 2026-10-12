"""User-run headless comparison. No training, video output, or controller access."""
import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from c1_tracking import overlap
from c1_tracking_advanced import AdvancedScreenTracker, AppearanceEncoder, CAMERA_PROFILES, global_assignment
from demo_c1_video_tracking import ROOT, SOURCE, WEIGHTS, APPEARANCE_WEIGHTS, load_clip, read_exact, prediction_rows
from fps_recorder.utils import resolve_executable

PREDICT_ARGS = dict(imgsz=1280, rect=False, device=0, quantize=16,
                    conf=.05, iou=.5, max_det=100, verbose=False, nms=None)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_engine(path, source_weights=None):
    manifest = json.loads((path.parent / 'export_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status') != 'exported' or not manifest.get('original_unchanged')
            or manifest['source_sha256'] != sha256(source_weights if source_weights is not None else WEIGHTS)
            or manifest['engine_sha256'] != sha256(path)):
        raise ValueError('候选引擎与原权重/导出回执不一致，停止比较。')
    required = dict(imgsz=1280, batch=1, dynamic=False, nms=None, max_det=100)
    if manifest['export_args'].get('nms') is not None:
        raise ValueError('旧候选切换了检测分支（nms=False）。请先用修正后的导出脚本重新导出，再测速。')
    if any(manifest['export_args'].get(key) != value for key, value in required.items()):
        raise ValueError('候选导出参数不符合本轮对比条件。')
    if manifest['export_args'].get('quantize') not in (16, 32):
        raise ValueError('本轮仅比较FP16/FP32候选。')
    return manifest


def latest_engine():
    for folder in sorted((ROOT / 'runs/c1_tensorrt').glob('*'), reverse=True):
        if (folder / 'detector.engine').is_file() and (folder / 'export_manifest.json').is_file():
            data = json.loads((folder / 'export_manifest.json').read_text(encoding='utf-8'))
            if data.get('status') == 'exported':
                return folder / 'detector.engine'
    raise FileNotFoundError('没有导出成功的候选引擎。')


def distribution(values):
    return dict(count=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95))) if values else dict(count=0, median=None, p95=None)


def compare_detections(old, new):
    """Agreement only, NOT accuracy: no ground-truth labels used here."""
    scores = [[overlap(a['box'], b['box']) if a['class'] == b['class']
               and overlap(a['box'], b['box']) >= .5 else None for b in new] for a in old]
    pairs = global_assignment(scores)
    used_old = {a for a, _ in pairs}; used_new = {b for _, b in pairs}
    remaining_old = [i for i in range(len(old)) if i not in used_old]
    remaining_new = [i for i in range(len(new)) if i not in used_new]
    conflicts = global_assignment([[overlap(old[a]['box'], new[b]['box'])
                                   if old[a]['class'] != new[b]['class'] and overlap(old[a]['box'], new[b]['box']) >= .5
                                   else None for b in remaining_new] for a in remaining_old])
    return dict(pt_boxes=len(old), engine_boxes=len(new), same_class_matches=len(pairs),
                pt_unmatched=len(old)-len(pairs), engine_unmatched=len(new)-len(pairs),
                spatial_class_disagreements=len(conflicts),
                matched_ious=[overlap(old[a]['box'], new[b]['box']) for a, b in pairs],
                matched_score_deltas=[abs(old[a]['score']-new[b]['score']) for a, b in pairs],
                matched_box_max_delta_px=[max(abs(x-y) for x, y in zip(old[a]['box'], new[b]['box'])) for a, b in pairs])


def run_backend(name, weights, path, destination, appearance, profile, warmup, precision=16):
    meta, frames = load_clip(path)
    width, height = meta['capture']['width'], meta['capture']['height']
    if (width, height) != (1920, 1080):
        raise ValueError('本轮要求1920×1080录制片段。')
    model = YOLO(str(weights), task='detect')
    predict_args = dict(PREDICT_ARGS, quantize=precision)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    print(f'{name}：预热 {warmup} 次；预热不计入测速。', flush=True)
    for _ in range(warmup):
        model.predict(blank, **predict_args)
    torch.cuda.synchronize()
    if model.names != {0: 'enemy', 1: 'teammate'}:
        raise ValueError('检测类别不一致。')
    runtime_end2end = bool(model.predictor.model.end2end)
    if runtime_end2end:
        raise ValueError('检测运行时仍为NMS-free分支，不能与现有模型公平比较。')
    tracker = AdvancedScreenTracker(max_lost_s=.2, memory_s=.8, appearance=True,
                                    camera_width=profile['width'], max_corners=profile['max_corners'])
    timings = {key: [] for key in ('detection', 'appearance', 'tracking', 'camera', 'visual_pipeline')}
    counts = Counter(); confirmed = set(); records = []; decoder = None
    with tempfile.TemporaryFile() as error_log, (destination / f'{name}_frames.jsonl').open('w', encoding='utf-8') as log:
        try:
            decoder = subprocess.Popen([resolve_executable('ffmpeg'), '-v', 'error', '-threads', '2',
                '-i', str(path / 'video.mkv'), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=error_log)
            for index, row in enumerate(frames):
                raw = read_exact(decoder.stdout, width*height*3)
                if len(raw) != width*height*3:
                    raise ValueError(f'第{index}帧解码提前结束。')
                rgb = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                timestamp = int(row['source_present_ns'])
                # A repeated presentation is not a new observation for tracker/votes.
                if tracker.timestamp == timestamp:
                    counts['reused_timestamp_frames'] += 1
                    continue
                torch.cuda.synchronize()
                begin = time.perf_counter()
                result = model.predict(rgb[:, :, ::-1].copy(), **predict_args)[0]
                detections = prediction_rows(result)
                torch.cuda.synchronize()
                after_detection = time.perf_counter()
                features = appearance.extract(rgb, detections)
                torch.cuda.synchronize()
                after_appearance = time.perf_counter()
                cv2.setRNGSeed(123)  # same RANSAC seed for both backends
                tracks = tracker.update_frame(timestamp, detections, rgb, features)
                end = time.perf_counter()
                sample = dict(detection=(after_detection-begin)*1000,
                              appearance=(after_appearance-after_detection)*1000,
                              tracking=(end-after_appearance)*1000,
                              camera=tracker.diagnostics['tracking_timings_ms']['camera'],
                              visual_pipeline=(end-begin)*1000)
                # Preserve the entire native-time trajectory, but exclude the first
                # 30 unique observations from steady-state latency distributions.
                if len(records) >= 30:
                    for key, value in sample.items(): timings[key].append(value)
                counts.update(tracker.events)
                counts['unassociated_boxes'] += sum(t['state'] == 'unassociated' for t in tracks)
                confirmed.update(t['track_id'] for t in tracks if t.get('confirmed') and t['track_id'] is not None)
                record = dict(frame_id=index, source_present_ns=timestamp, detections=detections,
                              tracks=tracks, timings_ms=sample, events=tracker.events)
                log.write(json.dumps(record, ensure_ascii=False)+'\n')  # outside timing
                records.append(dict(frame_id=index, source_present_ns=timestamp, detections=detections))
                if (index+1) % 240 == 0:
                    print(f'{name}：{index+1}/{len(frames)}', flush=True)
            if decoder.stdout.read(1): raise ValueError('视频含多余帧。')
            decoder.stdout.close()
            if decoder.wait(timeout=30):
                error_log.seek(0); raise RuntimeError(error_log.read().decode(errors='replace'))
        finally:
            if decoder is not None and decoder.poll() is None:
                decoder.kill(); decoder.wait(timeout=10)
    if not timings['visual_pipeline']:
        raise ValueError('有效帧太少，无法稳态测速。')
    summary = dict(model=str(weights), predict_args=predict_args, runtime_end2end=runtime_end2end,
                   timings_ms={k: distribution(v) for k, v in timings.items()},
                   counts=dict(counts), confirmed_ids=len(confirmed), unique_frames=len(records))
    del model, result
    gc.collect(); torch.cuda.empty_cache()
    return summary, records


def main():
    parser = argparse.ArgumentParser(description='无绘框/无视频编码的 PT 与 TensorRT 检测＋跟踪比较')
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--episode', type=int, default=35)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--tracking-profile', choices=list(CAMERA_PROFILES), default='threads')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--order', choices=['pt-first', 'engine-first'], default='pt-first')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.warmup < 10: parser.error('warmup 至少10次')
    engine = (args.engine or latest_engine()).resolve()
    manifest = verify_engine(engine)
    path = args.source / f'episode_{args.episode:06d}'
    _, frames = load_clip(path)
    if len({r['source_present_ns'] for r in frames}) <= 30: raise ValueError('片段不足31个独立时间戳。')
    if not APPEARANCE_WEIGHTS.is_file(): raise FileNotFoundError(APPEARANCE_WEIGHTS)
    resolve_executable('ffmpeg')
    print(f'片段：{path}，{len(frames)}帧\n候选：{engine}\n导出精度：FP{manifest["export_args"]["quantize"]}\n配置：{args.tracking_profile}', flush=True)
    if args.check_only: return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA不可用。')
    profile = CAMERA_PROFILES[args.tracking_profile]
    cv2.setNumThreads(profile['opencv_threads'])
    output = ROOT / 'runs/c1_tensorrt_benchmark' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    appearance = AppearanceEncoder(APPEARANCE_WEIGHTS)
    # Warm the unchanged appearance model separately; real crops still vary in count.
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(args.warmup): appearance.extract(dummy, [dict(box=[500, 200, 700, 700])])
    summaries = {}; records = {}
    order = ['pt', 'engine'] if args.order == 'pt-first' else ['engine', 'pt']
    for backend in order:
        summaries[backend], records[backend] = run_backend(backend, WEIGHTS if backend == 'pt' else engine,
                                                          path, output, appearance, profile, args.warmup,
                                                          precision=16 if backend == 'pt' else manifest['export_args']['quantize'])
    if [(r['frame_id'], r['source_present_ns']) for r in records['pt']] != [(r['frame_id'], r['source_present_ns']) for r in records['engine']]:
        raise ValueError('两个后端的原生时间戳序列不一致。')
    totals = Counter(); ious = []; score_deltas = []; box_deltas = []; changed = []
    for a, b in zip(records['pt'], records['engine']):
        comparison = compare_detections(a['detections'], b['detections'])
        for key in ('pt_boxes', 'engine_boxes', 'same_class_matches', 'pt_unmatched', 'engine_unmatched', 'spatial_class_disagreements'):
            totals[key] += comparison[key]
        ious.extend(comparison['matched_ious']); score_deltas.extend(comparison['matched_score_deltas'])
        box_deltas.extend(comparison['matched_box_max_delta_px'])
        if comparison['pt_unmatched'] or comparison['engine_unmatched'] or any(v < .9 for v in comparison['matched_ious']):
            changed.append(dict(frame_id=a['frame_id'], source_present_ns=a['source_present_ns'], **comparison))
    comparison = dict(totals=dict(totals), changed_frames=len(changed),
                      matched_iou=distribution(ious), score_delta=distribution(score_deltas),
                      box_max_delta_px=distribution(box_deltas))
    report = dict(stage='Stage 1 — Local Combat Policy', source=str(path), backend_order=order,
                  predict_args=PREDICT_ARGS, camera_profile=profile, warmup=args.warmup,
                  excluded_initial_unique_frames=30, export_manifest=manifest,
                  appearance_sha256=sha256(APPEARANCE_WEIGHTS), results=summaries,
                  detection_agreement=comparison, promoted=False, controller_output=False,
                  note='Wall latency includes RGB-to-BGR copy, detection preprocessing/transfer/postprocess, appearance and tracking. '
                       'Excludes video decode, drawing, encoding and log writing. Not pure forward or capture-to-action latency. '
                       'Single sequential comparison; hardware load/order may affect results. Agreement is not ground-truth accuracy.')
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'changed_frames.json').write_text(json.dumps(changed, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# TensorRT 视觉处理对比', '', '| 项目 | PT 中位数 / p95 | TensorRT 中位数 / p95 |', '| --- | --- | --- |']
    for key, label in [('detection', '检测调用'), ('appearance', '外观特征'), ('camera', '镜头补偿'), ('visual_pipeline', '检测＋跟踪总计')]:
        left = summaries['pt']['timings_ms'][key]; right = summaries['engine']['timings_ms'][key]
        lines.append(f"| {label} | {left['median']:.2f} / {left['p95']:.2f} ms | {right['median']:.2f} / {right['p95']:.2f} ms |")
    lines += ['', f'同类框 IoU≥0.5 一对一匹配：{totals["same_class_matches"]}。',
              f'PT未匹配框：{totals["pt_unmatched"]}；引擎未匹配框：{totals["engine_unmatched"]}。',
              f'空间重叠但类别不同的候选对：{totals["spatial_class_disagreements"]}；需核查变化帧：{len(changed)}。',
              '', '未匹配不等于误检/漏检；原模型不是标注真值，阈值附近的精度差异也可能改变输出。',
              '本报告只比较视觉处理，不含采集、动作决策或手柄输出；没有自动切换模型。',
              '单次顺序测试不能证明长期延迟稳定性。两个后端均保存逐帧数值日志，未生成视频。']
    (output / '对比报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'结果目录：{output}', flush=True)


if __name__ == '__main__': main()
