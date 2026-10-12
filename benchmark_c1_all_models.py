"""User-run single-frame FP16 latency, optional native-episode detection+tracking.

No training, capture service, rendered video, or controller output.
Models are sequential, multiple randomized blocks expose order variability.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/"c1_all_model_comparison.json"
PREDICT_ARGS=dict(imgsz=1280,rect=False,device=0,quantize=16,conf=.05,iou=.5,max_det=100,verbose=False,nms=None)


def distribution(values):
    if not values or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite timing samples")
    return {"count":len(values),"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),
            "min_ms":min(values),"max_ms":max(values)}


def load_models(config):
    rows=json.loads(config.read_text(encoding="utf-8"))["models"]
    if not rows or len({r["key"] for r in rows})!=len(rows):
        raise ValueError("Empty/duplicate model keys")
    joint_settings=[]
    full60_settings=[]
    for r in rows:
        if not r["key"].replace("_","").isalnum():
            raise ValueError("Model keys must be letters, numbers and underscores")
        p=Path(r["weights"])
        r["weights"]=(p if p.is_absolute() else ROOT/p).resolve()
        if not r["weights"].is_file():
            raise FileNotFoundError(f"模型尚未准备好，或实际目录名不同：{r['key']} {r['weights']}")
        checkpoint=torch.load(r["weights"],map_location="cpu",weights_only=False)
        if checkpoint.get("epoch")!=-1 or checkpoint.get("optimizer") is not None:
            raise ValueError(f"{r['key']}: checkpoint not finalized; wait for training completion")
        model=checkpoint.get("ema") or checkpoint.get("model")
        if getattr(model,"names",None)!={0:"enemy",1:"teammate"}:
            raise ValueError(f"{r['key']}: wrong classes")
        r["sha256"]=sha(r["weights"])
        scales={"joint_n":"n","joint_s":"s","joint_m":"m","joint_n_full60":"n"}
        if r["key"] in scales:
            receipt=json.loads((r["weights"].parent.parent/"data_receipt.json").read_text(encoding="utf-8"))
            initial=Path(receipt["initial_checkpoint"])
            if initial.name!=f"yolo26{scales[r['key']]}.pt" or sha(initial)!=receipt["initial_checkpoint_sha256"]:
                raise ValueError("Scale/initial checkpoint receipt differs")
            if receipt["manifest_sha256"]!=sha(ROOT/"c1_yolo_dataset/day3_v4_targeted_v1/source_manifest.json"):
                raise ValueError("Joint-model training data differs")
            settings={k:v for k,v in receipt["settings"].items() if k!="name"}
            if r["key"]=="joint_n_full60":
                if settings.get("epochs")!=60 or settings.get("patience")!=0:
                    raise ValueError("Full60 experiment must use epochs=60, patience=0")
                full60_settings.append(settings)
            else:
                joint_settings.append(settings)
        del model,checkpoint
    if joint_settings and any(settings!=joint_settings[0] for settings in joint_settings):
        raise ValueError("Joint n/s/m training settings differ; inspect before claiming scale-only comparison")
    if joint_settings and any(settings!={**joint_settings[0],"patience":0} for settings in full60_settings):
        raise ValueError("Full60 experiment differs beyond early stopping")
    return rows


def load_images():
    manifest=ROOT/"c1_yolo_dataset/day3_v1_normalized/source_manifest.json"
    records=[r for r in json.loads(manifest.read_text(encoding="utf-8")) if r["eligible"]]
    if not records:
        raise ValueError("No timing images")
    records=[records[int(i)] for i in np.linspace(0,len(records)-1,min(27,len(records)),dtype=int)]
    rgbs=[]
    for r in records:
        path=Path(r["normalized_image"])
        if sha(path)!=r["image_sha256"]:
            raise ValueError("Timing input image changed")
        with Image.open(path) as image:
            if image.size!=(1920,1080):
                raise ValueError("Timing inputs must match 1920x1080 recorder output")
            rgbs.append(np.array(image.convert("RGB")))
    if not rgbs:
        raise ValueError("No timing images")
    return rgbs,sha(manifest)


def call(model,rgb):
    # Wall boundary starts before RGB->BGR and finishes after CPU detection copies.
    torch.cuda.synchronize()
    start=time.perf_counter()
    bgr=np.ascontiguousarray(rgb[:,:,::-1])
    result=model.predict(bgr,**PREDICT_ARGS)[0]
    _=(result.boxes.xyxy.cpu(),result.boxes.conf.cpu(),result.boxes.cls.cpu())
    torch.cuda.synchronize()
    wall=(time.perf_counter()-start)*1000
    forward=float(result.speed["inference"])
    return forward,wall


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--warmup",type=int,default=50)
    parser.add_argument("--iterations",type=int,default=300)
    parser.add_argument("--blocks",type=int,default=3)
    parser.add_argument("--episode",type=Path,help="Optional complete recorder episode for native-time detection+tracking")
    args=parser.parse_args()
    if args.warmup<30 or args.iterations<100 or args.blocks<2:
        parser.error("warmup>=30, iterations>=100, blocks>=2 required")
    models=load_models(args.config)
    rgbs,image_manifest_sha=load_images()
    if args.episode:
        from demo_c1_video_tracking import load_clip,APPEARANCE_WEIGHTS
        _,frames=load_clip(args.episode)
        if len(frames)<=30 or not APPEARANCE_WEIGHTS.is_file():
            raise ValueError("Episode too short or appearance checkpoint unavailable")
    if args.check_only:
        print(f"检查通过：{len(models)}个已完成训练模型，{len(rgbs)}张统一输入；尚未测速。",flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-yolo CUDA environment")
    output=ROOT/"runs/c1_detection"/("all_model_latency_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    import ultralytics
    config={"models":[{**r,"weights":str(r["weights"])} for r in models],"predict_args":PREDICT_ARGS,"batch":1,
            "warmup":args.warmup,"iterations_per_block":args.iterations,"blocks":args.blocks,
            "image_manifest_sha256":image_manifest_sha,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"ultralytics":ultralytics.__version__,
            "forward_boundary":"Ultralytics synchronized inference profiler: AutoBackend model-call phase, no preprocess/postprocess",
            "predictor_boundary":"Synchronized wall: decoded CPU RGB -> CPU boxes/classes/scores, includes format conversion, preprocessing, transfers, model, NMS/postprocess",
            "excludes":"capture, image decoding, drawing/encoding, tracking (unless --episode), controller output",
            "episode":str(args.episode.resolve()) if args.episode else None,"controller_output":False}
    (output/"run_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    samples={r["key"]:{"forward":[],"predictor":[]} for r in models}
    details=[];blocks=[];orders=[]
    rng=random.Random(20261009)
    for block in range(args.blocks):
        order=models.copy();rng.shuffle(order);orders.append([r["key"] for r in order])
        for r in order:
            print(f"第{block+1}/{args.blocks}组：{r['key']}，预热{args.warmup}次，测速{args.iterations}次。",flush=True)
            model=YOLO(str(r["weights"]))
            for i in range(args.warmup):call(model,rgbs[i%len(rgbs)])
            if bool(model.predictor.model.end2end) or not model.predictor.model.fp16:
                raise ValueError("Wrong runtime branch/dtype; comparison aborted")
            forward,wall=[],[]
            for i in range(args.iterations):
                f,w=call(model,rgbs[i%len(rgbs)])
                forward.append(f);wall.append(w)
                details.append({"model":r["key"],"block":block+1,"sample":i+1,"image_index":i%len(rgbs),"forward_ms":f,"predictor_ms":w})
            samples[r["key"]]["forward"].extend(forward);samples[r["key"]]["predictor"].extend(wall)
            blocks.append({"model":r["key"],"block":block+1,"forward":distribution(forward),"predictor":distribution(wall)})
            del model;gc.collect();torch.cuda.empty_cache()
    summary={key:{phase:distribution(values) for phase,values in phases.items()} for key,phases in samples.items()}
    visual=[]
    if args.episode:
        from benchmark_c1_tensorrt import run_backend,AppearanceEncoder,CAMERA_PROFILES,APPEARANCE_WEIGHTS
        import cv2
        cv2.setNumThreads(CAMERA_PROFILES["threads"]["opencv_threads"])
        appearance=AppearanceEncoder(APPEARANCE_WEIGHTS)
        dummy=np.zeros((1080,1920,3),dtype=np.uint8)
        for _ in range(args.warmup):appearance.extract(dummy,[{"box":[500,200,700,700]}])
        for block,order in enumerate(orders,1):
            for key in order:
                r=next(r for r in models if r["key"]==key)
                print(f"检测+跟踪第{block}/{args.blocks}组：{key}",flush=True)
                name=key+f"_block{block}"
                data,_=run_backend(name,r["weights"],args.episode,output,appearance,CAMERA_PROFILES["threads"],args.warmup)
                visual.append({"model":key,"block":block,**data})
        (output/"visual_pipeline_blocks.json").write_text(json.dumps(visual,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"samples.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    (output/"summary.json").write_text(json.dumps({"models":summary,"blocks":blocks,"model_orders":orders},ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 所有候选单帧延迟对比","","batch=1、FP16、1280方形输入、相同NMS检测分支；每模型分组随机顺序测试。", "",
           "|模型|模型调用median ms|模型调用p95 ms|完整检测median ms|完整检测p95 ms|","|---|---:|---:|---:|---:|"]
    for key,m in summary.items():lines.append(f"|{key}|{m['forward']['median_ms']:.2f}|{m['forward']['p95_ms']:.2f}|{m['predictor']['median_ms']:.2f}|{m['predictor']['p95_ms']:.2f}|")
    lines += ["","模型调用计时由本地Ultralytics同步前向阶段提供，包含模型调用开销；不是纯CUDA kernel时间。完整检测不包含采集、追踪或控制。",
              "每组分布见summary.json，原始逐帧样本见samples.csv。不要把各阶段p95相加。统一使用27张1920×1080本机录制图片；只测延迟，不把训练图片用于泛化准确率结论。",
              "这是延迟比较，不代表识别准确率或capture-to-action。测速时需结束所有GPU训练及其他GPU负载。"]
    if visual:
        lines += ["","## 同一录制片段的检测+跟踪总延迟","","|模型|测试组|视觉总median ms|视觉总p95 ms|","|---|---:|---:|---:|"]
        for r in visual:
            t=r['timings_ms']['visual_pipeline'];lines.append(f"|{r['model']}|{r['block']}|{t['median']:.2f}|{t['p95']:.2f}|")
        lines.append("此阶段使用同一原生帧时间序列和threads跟踪配置，单独记录关联/类别冲突诊断；不含视频解码、采集和控制器。")
    (output/"延迟对比报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"结果：{output}",flush=True)


if __name__=="__main__":main()
