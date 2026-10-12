"""User-run paired conversion agreement and predict latency on 60 saved images."""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO
from benchmark_c1_tensorrt import verify_engine, compare_detections, distribution
from demo_c1_video_tracking import prediction_rows
from demo_c1_day4_supplement_tracking import WEIGHTS,EXPECTED_SHA
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT/'runs/c1_tensorrt/20261010_201904_287394/detector.engine'
ARGS=dict(imgsz=1280,rect=False,device=0,quantize=16,conf=.10,iou=.5,max_det=100,verbose=False,nms=None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if sha(WEIGHTS)!=EXPECTED_SHA:raise ValueError('PT weight changed')
    manifest=verify_engine(ENGINE,source_weights=WEIGHTS)
    rows=json.loads((ROOT/'c1_annotations/day4_supplement_v1/manifest.json').read_text(encoding='utf-8'))
    for r in rows:
        if sha(Path(r['image_path']))!=r['image_sha256']:raise ValueError('Prepared image changed')
    if args.check_only:
        print('检查通过，未运行对比。');return
    out=ROOT/'runs/c1_day4_engine_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    models={'pt':YOLO(str(WEIGHTS),task='detect'),'engine':YOLO(str(ENGINE),task='detect')}
    blank=np.zeros((1080,1920,3),np.uint8)
    for model in models.values():
        for _ in range(10):model.predict(blank,**ARGS)
        if model.predictor.model.end2end or model.names!={0:'enemy',1:'teammate'}:raise ValueError('Runtime branch/classes differ')
    values={'pt':[],'engine':[]};totals=Counter();details=[]
    for i,r in enumerate(rows):
        image=cv2.imread(r['image_path'])
        predictions={}
        # Alternate paired order, same loaded pixels. Excludes disk reads.
        for backend in (['pt','engine'] if i%2==0 else ['engine','pt']):
            start=time.perf_counter()
            predictions[backend]=prediction_rows(models[backend].predict(image,**ARGS)[0])
            values[backend].append((time.perf_counter()-start)*1000)
        comp=compare_detections(predictions['pt'],predictions['engine'])
        for key in ['pt_boxes','engine_boxes','same_class_matches','pt_unmatched','engine_unmatched','spatial_class_disagreements']:
            totals[key]+=comp[key]
        details.append(dict(image=r['image_path'],comparison=comp,**predictions))
        if (i+1)%20==0:print(f'同图对比{i+1}/{len(rows)}',flush=True)
    report=dict(source_weights_sha256=EXPECTED_SHA,engine_sha256=manifest['engine_sha256'],
        images=len(rows),predict_args=ARGS,predict_call_ms={k:distribution(v) for k,v in values.items()},
        agreement_totals=dict(totals),controller_output=False,promoted=False,
        scope='Conversion agreement on annotated supplement images, not accuracy/generalization evaluation. Paired alternating order; excludes game load/capture/tracking/display; not pure forward latency.')
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'frames.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('结果：',out)


if __name__=='__main__':main()
