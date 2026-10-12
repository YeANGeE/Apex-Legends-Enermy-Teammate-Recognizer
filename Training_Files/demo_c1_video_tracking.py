"""User-run offline Stage 1 tracking demo. No controller access or services.

Defaults to five existing held-out clips, tracking-only video.
Native source timestamps drive tracking, not frame/FPS or video PTS.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO

from c1_tracking import ScreenTracker
from c1_tracking_advanced import AdvancedScreenTracker, AppearanceEncoder, CAMERA_PROFILES
from fps_recorder.utils import resolve_executable

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY2_VAL')
WEIGHTS=ROOT/'runs/c1_yolo/day1_yolo26n_1280_v3_enemy_teammate/weights/best.pt'
APPEARANCE_WEIGHTS=ROOT/'weights/yolo26n-cls.pt'
NAMES={0:'敌人',1:'队友',None:'不确定'}
DETECTION_CONFIDENCE=.10  # 检测过滤阈值；推理、预热与结果记录共用。


def load_clip(path):
    meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    with (path/'frames.csv').open(newline='') as f:frames=list(csv.DictReader(f))
    if meta['recording_status']!='complete' or meta['encoder_queue_overflow'] or len(frames)!=meta['frame_count']:
        raise ValueError(f'Incomplete recording: {path}')
    times=[int(r['source_present_ns']) for r in frames]
    if not times or any(b<a for a,b in zip(times,times[1:])) or [int(r['frame_id']) for r in frames]!=list(range(len(frames))):
        raise ValueError('Nonmonotonic timestamp or encoded-frame index mismatch')
    return meta,frames


def draw_box(draw,b,color,label,dashed=False):
    x1,y1,x2,y2=b
    x1=max(0,min(1919,x1));x2=max(0,min(1919,x2))
    y1=max(0,min(1079,y1));y2=max(0,min(1079,y2))
    if x2<=x1 or y2<=y1:return
    if dashed:
        for x in range(int(x1),int(x2),18):
            draw.line((x,y1,min(x+9,x2),y1),fill=color,width=3)
            draw.line((x,y2,min(x+9,x2),y2),fill=color,width=3)
        for y in range(int(y1),int(y2),18):
            draw.line((x1,y,x1,min(y+9,y2)),fill=color,width=3)
            draw.line((x2,y,x2,min(y+9,y2)),fill=color,width=3)
    else:draw.rectangle((x1,y1,x2,y2),outline=color,width=3)
    draw.text((x1,max(80,y1-26)),label,font=FONT,fill=color,stroke_width=1,stroke_fill='black')


def render(frame,predictions,tracks,header,show_predictions=False,split_view=False,show_unassociated=False,target_selection=None):
    left=Image.fromarray(frame);right=left.copy()
    a,b=ImageDraw.Draw(left),ImageDraw.Draw(right)
    if split_view:
        for d in predictions:
            draw_box(a,d['box'],'#ff6868' if d['class']==0 else '#50bfff',f"{NAMES[d['class']]} {d['score']:.2f}")
    for tr in tracks:
        state=tr['state'];cls=tr['class_id']
        if state=='remembered' or (state=='tentative' and not show_unassociated) or (state=='predicted' and (not show_predictions or not tr.get('confirmed',True))):
            continue
        color='#ffce55' if state=='predicted' else '#aaaaaa' if cls is None else '#ff6868' if cls==0 else '#50bfff'
        label=('未关联 ' if state=='unassociated' else '待确认 ' if state=='tentative' else f"#{tr['track_id']} ")
        label+=f"仅预测 {tr['age_ms']:.0f}ms" if state=='predicted' else f"{NAMES[cls]} 检测"
        draw_box(b,tr['box'],color,label,state=='predicted')
    title_right=('跟踪：黄色虚线仅预测，不代表真实检出' if show_predictions else
                 '跟踪：灰框是当前检测但身份/编号待确认；预测不绘制' if show_unassociated else
                 '跟踪：仅显示已确认的当前检测；预测保留在日志，不绘制')
    for draw,title in ((a,'原始检测：每帧独立'),(b,title_right)):
        draw.rectangle((0,0,1920,76),fill='black')
        draw.text((8,4),title,font=FONT,fill='white')
        draw.text((8,40),header,font=FONT,fill='white')
    memory=[tr for tr in tracks if tr['state'] in ('predicted','remembered') and tr.get('confirmed',False)]
    if memory:
        text='暂不可见，保留记忆：'+ ' / '.join(f"#{tr['track_id']} {tr['age_ms']:.0f}ms" for tr in memory[:5])
        b.rectangle((0,80,1000,115),fill='black');b.text((8,83),text,font=FONT,fill='#ffce55')
    if target_selection is not None:
        from c1_target_selection import REASONS
        selected=target_selection['selected']
        if selected is not None:
            draw_box(b,selected['box'],'#00ff88',f"当前选择 #{selected['track_id']} 敌人")
        text=f"目标选择：{'无' if selected is None else '#'+str(selected['track_id'])} · {REASONS[target_selection['reason']]} · 候选{target_selection['candidate_count']}"
        b.rectangle((0,120,1920,158),fill='black');b.text((8,125),text,font=FONT,fill='#00ff88')
    if not split_view:
        return right
    canvas=Image.new('RGB',(3840,1080));canvas.paste(left);canvas.paste(right,(1920,0))
    return canvas


def read_exact(pipe,size):
    data=bytearray()
    while len(data)<size:
        part=pipe.read(size-len(data))
        if not part:break
        data.extend(part)
    return bytes(data)


def prediction_rows(result):
    """One device-to-host transfer instead of three separate tensor transfers."""
    data=result.boxes.data
    if data.ndim!=2 or data.shape[1]!=6:raise ValueError('Expected untracked xyxy/conf/class detection tensor')
    return [dict(box=row[:4],score=row[4],**{'class':int(row[5])}) for row in data.cpu().tolist()]


def run_clip(model,path,output,max_lost,video_encoder='h264_nvenc',show_predictions=False,split_view=False,memory_s=.8,
             tracker_mode='advanced',appearance_encoder=None,tracking_profile='threads',detector_weights=WEIGHTS,stable_geometry=False,group_duplicates=False,
             select_target=False,target_high_confidence=.25,target_large_area=.01):
    if tracker_mode=='advanced' and appearance_encoder is None:
        raise ValueError('Advanced mode requires an appearance encoder; use motion or baseline explicitly')
    profile=CAMERA_PROFILES[tracking_profile]
    cv2.setNumThreads(profile['opencv_threads'])
    meta,frames=load_clip(path)
    width,height=meta['capture']['width'],meta['capture']['height']
    if (width,height)!=(1920,1080):raise ValueError('Expected 1920x1080')
    ffmpeg=resolve_executable('ffmpeg');fps=meta['encoder']['output_fps']
    output_width=3840 if split_view else 1920
    destination=output/path.name;destination.mkdir()
    tracker=(ScreenTracker(max_lost_s=max_lost,memory_s=memory_s) if tracker_mode=='baseline' else
             AdvancedScreenTracker(max_lost_s=max_lost,memory_s=memory_s,appearance=tracker_mode=='advanced',
                                   camera_width=profile['width'],max_corners=profile['max_corners'],stable_geometry=stable_geometry,group_duplicates=group_duplicates))
    counts=Counter();latencies=[];previous_thumbnail=None;predictions=[];tracks=[];confirmed_ids=set()
    selector=None;selection=None
    if select_target:
        from c1_target_selection import EnemyTargetSelector
        selector=EnemyTargetSelector(high_confidence=target_high_confidence,large_area_ratio=target_large_area)
    camera_parts=('prepare','corners','flow_forward','flow_backward','filter','ransac')
    phases={name:[] for name in ('decode_read','appearance','tracking','camera','association_and_state',
                                'render','encode_submit','detection_and_tracking',*(f'camera_{name}' for name in camera_parts))}
    decoder=encoder=None
    started=time.perf_counter()
    with tempfile.TemporaryFile() as decode_log,tempfile.TemporaryFile() as encode_log,(destination/'tracks.jsonl').open('w',encoding='utf-8') as log:
        try:
            decoder=subprocess.Popen([ffmpeg,'-v','error','-threads','2','-i',str(path/'video.mkv'),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=decode_log)
            precision_args=(['-preset','p4','-tune','hq','-rc','vbr','-cq','18','-b:v','0']
                            if video_encoder=='h264_nvenc' else ['-threads','2','-preset','fast','-crf','18'])
            encoder=subprocess.Popen([ffmpeg,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{output_width}x1080','-r',str(fps),'-i','pipe:0','-an',
                '-filter_threads','2','-vf','format=nv12','-c:v',video_encoder,*precision_args,
                '-pix_fmt','yuv420p','-movflags','+faststart',str(destination/'跟随演示.mp4')],stdin=subprocess.PIPE,stderr=encode_log)
            for index,row in enumerate(frames):
                before=time.perf_counter()
                data=read_exact(decoder.stdout,width*height*3)
                phases['decode_read'].append((time.perf_counter()-before)*1000)
                if len(data)!=width*height*3:raise ValueError(f'Decode ended early at frame {index}')
                frame=np.frombuffer(data,dtype=np.uint8).reshape(height,width,3)
                timestamp=int(row['source_present_ns'])
                repeated=tracker.timestamp is not None and timestamp==tracker.timestamp
                if not repeated:
                    thumbnail=frame[::32,::32].astype(np.float32)
                    # Conservative scene/camera-jump heuristic; not a world model.
                    cut=previous_thumbnail is not None and float(np.mean(np.abs(thumbnail-previous_thumbnail)))/255>.25
                    # Preserve V2 only for explicit comparison. Brightness jumps
                    # (muzzle flash) are NOT sufficient proof of a scene cut.
                    if cut and tracker_mode=='baseline':tracker.reset();counts['scene_resets']+=1
                    previous_thumbnail=thumbnail
                    before=time.perf_counter()
                    # Ultralytics ndarray input is BGR, decoded pipe is RGB.
                    result=model.predict(frame[:,:,::-1].copy(),imgsz=1280,rect=False,device=0,quantize=16,conf=DETECTION_CONFIDENCE,iou=.5,max_det=100,verbose=False,nms=None)[0]
                    predictions=prediction_rows(result)
                    latencies.append((time.perf_counter()-before)*1000)
                    before=time.perf_counter()
                    features=appearance_encoder.extract(frame,predictions) if tracker_mode=='advanced' else None
                    phases['appearance'].append((time.perf_counter()-before)*1000)
                    before=time.perf_counter()
                    tracks=(tracker.update(timestamp,predictions) if tracker_mode=='baseline' else
                            tracker.update_frame(timestamp,predictions,frame,features))
                    phases['tracking'].append((time.perf_counter()-before)*1000)
                    diagnostic=getattr(tracker,'diagnostics',{})
                    for name in ('camera','association_and_state'):
                        phases[name].append(diagnostic.get('tracking_timings_ms',{}).get(name,0.))
                    for name in camera_parts:
                        phases[f'camera_{name}'].append(diagnostic.get('camera',{}).get('timings_ms',{}).get(name,0.))
                    phases['detection_and_tracking'].append(latencies[-1]+phases['appearance'][-1]+phases['tracking'][-1])
                    counts.update(tracker.events);counts['inferred_frames']+=1
                else:counts['reused_timestamp_frames']+=1
                if selector is not None and not repeated:
                    selection=selector.update(tracks,timestamp,width,height)
                    counts['target_'+selection['reason']]+=1
                counts['prediction_only_boxes']+=sum(tr['state']=='predicted' for tr in tracks)
                counts['memory_only_tracks']+=sum(tr['state']=='remembered' for tr in tracks)
                counts['unassociated_detection_boxes']+=sum(tr['state']=='unassociated' for tr in tracks)
                confirmed_ids.update(tr['track_id'] for tr in tracks if tr.get('confirmed') and tr['track_id'] is not None)
                mode_name={'advanced':'镜头＋外观','motion':'镜头补偿','baseline':'旧版基线'}[tracker_mode]
                header=f"{path.name} 帧{index} 源时间{(timestamp-int(frames[0]['source_present_ns']))/1e9:.2f}s {mode_name} · 无手柄输出"
                before=time.perf_counter()
                image=render(frame,predictions,tracks,header,show_predictions,split_view,tracker_mode!='baseline',selection)
                phases['render'].append((time.perf_counter()-before)*1000)
                before=time.perf_counter()
                encoder.stdin.write(image.tobytes())
                phases['encode_submit'].append((time.perf_counter()-before)*1000)
                log.write(json.dumps(dict(frame_id=index,source_present_ns=timestamp,reused_timestamp=repeated,detections=predictions,tracks=tracks,
                                         events=tracker.events if not repeated else [],
                                         association_diagnostics=getattr(tracker,'diagnostics',{}),target_selection=selection),ensure_ascii=False)+'\n')
                if index in (len(frames)//3,2*len(frames)//3):image.save(destination/f'预览_{index:06d}.jpg',quality=95,subsampling=0)
                if (index+1)%120==0:print(f'{path.name}: {index+1}/{len(frames)}',flush=True)
            if decoder.stdout.read(1):raise ValueError('Video has extra decoded frames')
            decoder.stdout.close();encoder.stdin.close()
            if decoder.wait(timeout=30) or encoder.wait(timeout=30):
                decode_log.seek(0);encode_log.seek(0)
                raise RuntimeError((decode_log.read()+encode_log.read()).decode(errors='replace'))
        finally:
            for process in (decoder,encoder):
                if process is not None and process.poll() is None:
                    process.kill();process.wait(timeout=10)
    summary=dict(stage='Stage 1 — Local Combat Policy',source=str(path),frames=len(frames),
        counts=dict(counts),prediction_call_median_ms=float(np.median(latencies)),prediction_call_p95_ms=float(np.percentile(latencies,95)),
        timing_note='Synchronous predict() wall time including preprocessing/transfer/postprocess, excluding decode/tracking/drawing/encoding; offline not capture-to-action latency',
        offline_elapsed_s=time.perf_counter()-started,model=str(detector_weights),detector_backend='tensorrt' if Path(detector_weights).suffix=='.engine' else 'pytorch',
        confidence=DETECTION_CONFIDENCE,max_lost_s=max_lost,memory_s=memory_s,target_selection_enabled=select_target,
        target_high_confidence=target_high_confidence,target_large_area_ratio=target_large_area,
        tracker_version=2 if tracker_mode=='baseline' else 6 if group_duplicates else 5 if stable_geometry else 4,
        stable_geometry=stable_geometry,group_duplicates=group_duplicates,tracker_mode=tracker_mode,ever_confirmed_track_count=len(confirmed_ids),
        appearance_model=str(getattr(appearance_encoder,'model_path',APPEARANCE_WEIGHTS)) if tracker_mode=='advanced' else None,
        tracking_profile=tracking_profile,camera_profile=profile,opencv_threads=cv2.getNumThreads(),
        controller_output=False,training_started=False,video_encoder=video_encoder,
        show_prediction_boxes=show_predictions,
        split_view=split_view,output_size=[output_width,1080],
        phase_timings_ms={name:dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),total=float(sum(values))) for name,values in phases.items()},
        phase_note='encode_submit includes RGB serialization and pipe backpressure, not isolated NVENC compute; decode_read includes pipe wait/copy. Phase p95 values must not be added to estimate total p95.',
        acceleration_note='YOLO CUDA; NVENC encoding when selected; decode/PIL drawing/RGB conversion remain CPU',
        warning='No tracking ground truth: ID continuity/identity not certified. Appearance encoder is not game-trained ReID; camera affine approximation can fail. No aim/recoil or world compensation.')
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"检测+跟踪：中位数 {summary['phase_timings_ms']['detection_and_tracking']['median']:.2f}ms，p95 {summary['phase_timings_ms']['detection_and_tracking']['p95']:.2f}ms；离线生成 {summary['offline_elapsed_s']:.1f}秒",flush=True)
    print(f"镜头补偿：{summary['phase_timings_ms']['camera']['median']:.2f}ms；关联与状态：{summary['phase_timings_ms']['association_and_state']['median']:.2f}ms；配置：{tracking_profile}",flush=True)
    print(f'完成：{destination / "跟随演示.mp4"}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',nargs='+',type=int,default=[6,8,9,21,35])
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--detector',type=Path,default=WEIGHTS,help='Default original PT; explicit validated TensorRT candidate is opt-in')
    parser.add_argument('--max-lost',type=float,default=.2)
    parser.add_argument('--memory',type=float,default=.8,help='Trajectory memory horizon; independent of extrapolation')
    parser.add_argument('--tracker',choices=['advanced','motion','baseline'],default='advanced',
                        help='Advanced: camera+global association+appearance; motion: no appearance; baseline: old V2')
    parser.add_argument('--appearance-model',type=Path,default=APPEARANCE_WEIGHTS,
                        help='Pretrained descriptor checkpoint; first user-run may download yolo26n-cls.pt')
    parser.add_argument('--tracking-profile',choices=list(CAMERA_PROFILES),default='threads',
                        help='threads: 4 CPU threads, unchanged camera quality; reference: old 32 threads; fast: experimental 480px/200 points')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--stable-geometry',action='store_true',help='Experimental V5 recent confirmed track area uses measured box, not shrinking predicted box')
    parser.add_argument('--group-duplicates',action='store_true',help='Experimental V6 duplicate position grouping; cross-class conflict stays uncertain')
    parser.add_argument('--show-predictions',action='store_true',help='Show yellow prediction boxes; otherwise log only')
    parser.add_argument('--split-view',action='store_true',help='Raw detection left, tracking right; default tracking only')
    parser.add_argument('--select-target',action='store_true',help='Enemy-only lock-and-switch visualization; no actions')
    parser.add_argument('--target-high-confidence',type=float,default=.25)
    parser.add_argument('--target-large-area',type=float,default=.01,help='Large-body threshold as image-area ratio')
    parser.add_argument('--encoder',choices=['h264_nvenc','libx264'],default='h264_nvenc',
                        help='Default NVIDIA hardware encoding; libx264 explicitly uses CPU')
    args=parser.parse_args(argv)
    if not .05<=args.target_high_confidence<=1 or not 0<args.target_large_area<=1:
        parser.error('Invalid target confidence/area thresholds')
    if not .01<=args.max_lost<=.5:parser.error('max-lost must be .01 to .5 seconds')
    if not args.max_lost<=args.memory<=2:parser.error('memory must be >= max-lost and <=2 seconds')
    if args.stable_geometry and args.tracker=='baseline':parser.error('stable-geometry requires advanced or motion tracker')
    if args.group_duplicates and args.tracker!='advanced':parser.error('group-duplicates requires advanced appearance tracker')
    paths=[args.source/f'episode_{n:06d}' for n in args.episodes]
    if len(set(paths))!=len(paths):raise ValueError('Duplicate episode selection')
    for path in paths:
        _,frames=load_clip(path);print(f'检查：{path}，{len(frames)}帧')
    args.detector=args.detector.resolve()
    if not args.detector.is_file() or not torch.cuda.is_available():raise RuntimeError('Missing detector weights/CUDA')
    if args.detector.suffix=='.engine':
        # Import only after demo module initialization (benchmark reuses demo helpers).
        from benchmark_c1_tensorrt import verify_engine
        manifest=verify_engine(args.detector)
        if manifest['export_args']['quantize']!=16:raise ValueError('This tracking regression expects FP16 candidate')
    elif args.detector.suffix!='.pt':raise ValueError('Expected .pt or .engine detector')
    print(f'检测器：{args.detector}；保留原候选框＋NMS路径',flush=True)
    resolve_executable('ffmpeg')
    if args.check_only:return
    cv2.setNumThreads(CAMERA_PROFILES[args.tracking_profile]['opencv_threads'])
    appearance_encoder=None
    if args.tracker=='advanced':
        args.appearance_model.parent.mkdir(parents=True,exist_ok=True)
        print(f'加载外观特征模型：{args.appearance_model}；首次可能下载。仅辅助编号，不改敌我分类。',flush=True)
        appearance_encoder=AppearanceEncoder(args.appearance_model)
    model=YOLO(str(args.detector),task='detect')
    if model.names!={0:'enemy',1:'teammate'}:raise ValueError('Checkpoint class mismatch')
    for _ in range(10):model.predict(np.zeros((1080,1920,3),dtype=np.uint8),imgsz=1280,rect=False,device=0,quantize=16,conf=DETECTION_CONFIDENCE,iou=.5,max_det=100,verbose=False,nms=None)
    if model.predictor.model.end2end:raise ValueError('Unexpected NMS-free head; regression must preserve original detection path')
    output=ROOT/'runs/c1_tracking'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    (output/'run_config.json').write_text(json.dumps(dict(weights=str(args.detector),weights_sha256=hashlib.sha256(args.detector.read_bytes()).hexdigest(),
        detector_backend='tensorrt' if args.detector.suffix=='.engine' else 'pytorch',runtime_end2end=False,
        source_clips=[str(p) for p in paths],controller_output=False,source_timestamps_only=True,
        tracker_version=2 if args.tracker=='baseline' else 6 if args.group_duplicates else 5 if args.stable_geometry else 4,
        stable_geometry=args.stable_geometry,group_duplicates=args.group_duplicates,tracker_mode=args.tracker,
        tracking_profile=args.tracking_profile,camera_profile=CAMERA_PROFILES[args.tracking_profile],
        appearance_model=str(args.appearance_model) if args.tracker=='advanced' else None,
        appearance_sha256=hashlib.sha256(args.appearance_model.read_bytes()).hexdigest() if appearance_encoder and args.appearance_model.is_file() else None,
        max_lost_s=args.max_lost,memory_s=args.memory,confidence=DETECTION_CONFIDENCE,video_encoder=args.encoder,split_view=args.split_view,show_predictions=args.show_predictions,
        target_selection_enabled=args.select_target,target_high_confidence=args.target_high_confidence,target_large_area_ratio=args.target_large_area),indent=2),encoding='utf-8')
    print(f'视频编码：{args.encoder}；YOLO：CUDA；解码与绘框：CPU',flush=True)
    print(f'跟踪模式：{args.tracker}；画面亮度跳变不再重置新版人物编号。',flush=True)
    print(f'面积门控：{"V5近期实测框参照（候选）" if args.stable_geometry else "V4原外推框参照"}',flush=True)
    print(f'重复位置分组：{args.group_duplicates}；类别冲突不作为敌人证据',flush=True)
    for path in paths:run_clip(model,path,output,args.max_lost,args.encoder,args.show_predictions,args.split_view,args.memory,args.tracker,appearance_encoder,args.tracking_profile,args.detector,args.stable_geometry,args.group_duplicates,
                              args.select_target,args.target_high_confidence,args.target_large_area)
    print(f'全部完成，结果目录：{output}')


FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
if __name__=='__main__':main()
