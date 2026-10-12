"""User-run CPU camera profiling using existing detections, no YOLO inference.

Runs each profile separately over the same complete clip for fair scheduling.
Native timestamps preserve causal camera history; the tail is evaluated without
resetting at its boundary. No services/controller/training/model changes.
"""
import argparse
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from c1_tracking_advanced import CAMERA_PROFILES,CameraCompensator
from demo_c1_video_tracking import ROOT,load_clip,read_exact
from fps_recorder.utils import resolve_executable


def statistics(values):
    return dict(count=len(values),median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values,95)) if values else None)


def latest_run(episode):
    for run in sorted((ROOT/'runs/c1_tracking').iterdir(),reverse=True):
        clip=run/f'episode_{episode:06d}'
        if run.is_dir() and (clip/'tracks.jsonl').is_file() and (clip/'summary.json').is_file():return run
    raise FileNotFoundError('No completed tracking log for requested episode')


def compare(reference,candidate,width,height):
    mismatch=0;differences=[]
    points=np.array([[0,0,1],[width,0,1],[width,height,1],[0,height,1],[width/2,height/2,1.]])
    for a,b in zip(reference,candidate):
        mismatch+=a['status']!=b['status']
        if a['matrix'] is not None and b['matrix'] is not None:
            deltas=points@(np.asarray(a['matrix'])-np.asarray(b['matrix'])).T
            differences.append(float(np.linalg.norm(deltas,axis=1).mean()))
    return dict(status_mismatch_frames=mismatch,mean_five_screen_points_displacement_px=statistics(differences))


def run_profile(path,rows,profile):
    meta,frames=load_clip(path)
    if len(rows)!=len(frames):raise ValueError('Log/video frame count differs')
    width,height=meta['capture']['width'],meta['capture']['height']
    config=CAMERA_PROFILES[profile];cv2.setNumThreads(config['opencv_threads'])
    camera=CameraCompensator(config['width'],config['max_corners'])
    times=[int(r['source_present_ns']) for r in frames]
    cutoff=times[0]+(times[-1]-times[0])/3
    results=[];statuses=Counter();timings={};tail_timings={};unique=0
    with tempfile.TemporaryFile() as error:
        process=subprocess.Popen([resolve_executable('ffmpeg'),'-v','error','-threads','2','-i',str(path/'video.mkv'),
            '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=error)
        try:
            previous=None
            for index,(row,t) in enumerate(zip(rows,times)):
                data=read_exact(process.stdout,width*height*3)
                if len(data)!=width*height*3:raise ValueError('Early video EOF')
                if row['frame_id']!=index or row['source_present_ns']!=t:raise ValueError('Log/native timestamp mismatch')
                if t==previous:continue
                frame=np.frombuffer(data,np.uint8).reshape(height,width,3)
                cv2.setRNGSeed(123)  # Same RANSAC random seed across profiles.
                _,info=camera.update(frame,row['detections'],t)
                previous=t;unique+=1;statuses[info['status']]+=1
                results.append(dict(timestamp_ns=t,status=info['status'],matrix=info['matrix']))
                # Exclude first 30 unique frames only from steady-state timings.
                # Quality counters still include the entire clip.
                if unique>30:
                    for phase,value in info['timings_ms'].items():
                        timings.setdefault(phase,[]).append(value)
                        if t>=cutoff:tail_timings.setdefault(phase,[]).append(value)
                if (index+1)%240==0:print(f'{profile}: {index+1}/{len(frames)}',flush=True)
            if process.stdout.read(1):raise ValueError('Extra decoded frames')
            process.stdout.close()
            if process.wait(timeout=30):
                error.seek(0);raise RuntimeError(error.read().decode(errors='replace'))
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
    return dict(profile=profile,config=config,status_counts=dict(statuses),unique_frames=unique,
                steady_timings_ms={k:statistics(v) for k,v in timings.items()},
                tail_two_thirds_timings_ms={k:statistics(v) for k,v in tail_timings.items()},
                tail_start_elapsed_s=(cutoff-times[0])/1e9),results,(width,height)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,default=35)
    parser.add_argument('--tracking-run',type=Path);args=parser.parse_args()
    run=args.tracking_run or latest_run(args.episode)
    clip=run/f'episode_{args.episode:06d}'
    source=Path(json.loads((clip/'summary.json').read_text(encoding='utf-8'))['source'])
    rows=[json.loads(s) for s in (clip/'tracks.jsonl').read_text(encoding='utf-8').splitlines()]
    if len({r['source_present_ns'] for r in rows})<=30:
        raise ValueError('Profiling requires more than 30 unique source timestamps')
    output=ROOT/'runs/c1_camera_benchmark'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True)
    print(f'仅CPU镜头补偿测速，不加载检测/外观模型。输入：{run}',flush=True)
    reports=[];reference=None
    for profile in CAMERA_PROFILES:
        report,estimates,size=run_profile(source,rows,profile)
        if reference is None:reference=estimates
        report['comparison_to_reference']=compare(reference,estimates,*size)
        reports.append(report)
        print(f"{profile}：镜头补偿中位 {report['steady_timings_ms']['total']['median']:.2f}ms，p95 {report['steady_timings_ms']['total']['p95']:.2f}ms",flush=True)
    payload=dict(stage='Stage 1 — Local Combat Policy',episode=args.episode,tracking_run=str(run),source=str(source),
        controller_output=False,training_started=False,model_inference=False,reports=reports,
        note='Camera-only CPU timings exclude decoding, detection, appearance, association, rendering and encoding. No automatic profile promotion; reference disagreement is not accuracy.')
    (output/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 镜头补偿配置测速','',f'输入：`{run}`','',
           '仅CPU镜头补偿，排除前30个唯一时间戳帧的耗时；质量状态计数包含全片。没有推理/训练/控制器输出。','',
           '|配置|全片稳态中位ms|p95 ms|后2/3中位ms|后2/3 p95 ms|拒绝帧|与参考状态不同帧|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        all_time=r['steady_timings_ms']['total'];tail=r['tail_two_thirds_timings_ms']['total']
        lines.append(f"|{r['profile']}|{all_time['median']:.2f}|{all_time['p95']:.2f}|{tail['median']:.2f}|{tail['p95']:.2f}|{r['status_counts'].get('rejected',0)}|{r['comparison_to_reference']['status_mismatch_frames']}|")
    lines+=['','threads保持640像素/350点，仅32→4线程；fast为480像素/200点，尚未验收。',
            '状态一致或矩阵接近不证明跟踪正确；拒绝减少也不自动证明更准确。需要完整视频回归与检测+跟踪整体耗时确认。']
    (output/'测速报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'完成：{output / "测速报告.md"}',flush=True)


if __name__=='__main__':main()
