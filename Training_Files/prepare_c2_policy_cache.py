"""User-run frozen C1 observations + causal human actions; no policy training."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from c2_policy import detection_features,ACTION_SCHEMA,ACTION_ORDER
from prepare_c2_recording_manifest import OUTPUT
from benchmark_c1_day4_engine import ENGINE
from demo_c1_day4_supplement_tracking import WEIGHTS
from benchmark_c1_tensorrt import verify_engine
from demo_c1_video_tracking import prediction_rows,read_exact
from fps_recorder.utils import resolve_executable
from normalize_c1_day3_export import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all',action='store_true',help='Prepare all train/val clips, keeping test/diagnostic unused')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    manifest=OUTPUT/'recording_manifest.json'
    roster=json.loads(manifest.read_text(encoding='utf-8'))
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    if summary.get('action_schema')!=ACTION_SCHEMA:raise ValueError('Wrong action label schema')
    rows=[r for r in roster if r['split'] in ['train','val']] if args.all else [r for r in roster if r['key']==summary['smoke_clip']]
    engine_receipt=verify_engine(ENGINE,source_weights=WEIGHTS)
    provenance=dict(manifest_sha256=sha(manifest),engine_sha256=engine_receipt['engine_sha256'],
        feature_version='C1_top8_score_xywh_enemy_team_v1',image_size=[288,160],input_stream='gamepad.csv',
        action_schema=ACTION_SCHEMA,action_order=ACTION_ORDER)
    if args.check_only:
        print('将准备',len(rows),'段，未推理/训练。');return
    print(f'准备{len(rows)}段缓存：包含完整解码检查及冻结C1推理，完成之前不训练C2。',flush=True)
    model=YOLO(str(ENGINE),task='detect')
    predict_args=dict(imgsz=1280,rect=False,device=0,quantize=16,conf=.10,iou=.5,max_det=100,verbose=False,nms=None)
    model.predict(np.zeros((1080,1920,3),np.uint8),**predict_args)
    if model.predictor.model.end2end:raise ValueError('Wrong detector branch')
    for number,r in enumerate(rows,1):
        folder=OUTPUT/'cache'/r['key'];receipt=folder/'receipt.json'
        expected={**provenance,'key':r['key'],'source_files_sha256':r['files_sha256']}
        path=Path(r['episode_path'])
        for name,digest in r['files_sha256'].items():
            if sha(path/name)!=digest:raise ValueError('Original recording metadata/input changed')
        if receipt.exists():
            previous=json.loads(receipt.read_text(encoding='utf-8'))
            if previous['provenance']!=expected:raise ValueError('Existing cache provenance differs')
            for name,digest in previous['cache_files_sha256'].items():
                if sha(folder/name)!=digest:raise ValueError('Cached arrays changed')
            if sha(path/'video.mkv')!=previous['video_sha256']:raise ValueError('Original video changed')
            print(f'{number}/{len(rows)} {r["key"]}: 已核验，复用缓存',flush=True)
            continue
        if folder.exists():raise ValueError(f'Partial cache preserved for inspection: {folder}')
        wanted={s['frame_id']:s for s in r['samples']};images=[];features=[];actions=[];timestamps=[]
        with tempfile.TemporaryFile() as errors:
            process=subprocess.Popen([resolve_executable('ffmpeg'),'-v','error','-threads','2','-i',str(path/'video.mkv'),
                '-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,stderr=errors)
            try:
                for i in range(r['frames']):
                    data=read_exact(process.stdout,1920*1080*3)
                    if len(data)!=1920*1080*3:raise ValueError('Video decode ended before frame CSV')
                    if i not in wanted:continue
                    frame=np.frombuffer(data,np.uint8).reshape(1080,1920,3)
                    predictions=prediction_rows(model.predict(frame,**predict_args)[0])
                    images.append(cv2.cvtColor(cv2.resize(frame,(288,160),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
                    features.append(detection_features(predictions));actions.append(wanted[i]['action']);timestamps.append(wanted[i]['timestamp_ns'])
                if process.stdout.read(1):raise ValueError('Video contains extra frames')
                process.stdout.close()
                if process.wait(timeout=30):
                    errors.seek(0);raise RuntimeError(errors.read().decode(errors='replace'))
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=10)
        if len(images)!=len(r['samples']):raise ValueError('Missing selected source frames')
        folder.mkdir(parents=True)
        arrays=dict(images=np.asarray(images,np.uint8),features=np.asarray(features,np.float32),
                    actions=np.asarray(actions,np.float32),timestamps=np.asarray(timestamps,np.int64))
        for name,array in arrays.items():np.save(folder/(name+'.npy'),array,allow_pickle=False)
        receipt.write_text(json.dumps(dict(provenance=expected,video_sha256=sha(path/'video.mkv'),
            cache_files_sha256={name+'.npy':sha(folder/(name+'.npy')) for name in arrays},
            decoded_frames=r['frames'],samples=len(images),training_started=False),indent=2),encoding='utf-8')
        print(f'{number}/{len(rows)} {r["key"]}: {len(images)}个因果动作样本',flush=True)
    print('数据准备完成；未启动C2训练。')


if __name__=='__main__':main()
