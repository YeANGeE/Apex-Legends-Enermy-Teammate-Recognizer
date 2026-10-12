"""Prepare a small native-resolution review of candidate target-following segments.

Detector boxes are proposals, not verified intent. No training or controller output.
"""
import json
import csv
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from c2_policy import ACTION_SCHEMA
from demo_c1_video_tracking import read_exact
from fps_recorder.utils import resolve_executable
from normalize_c1_day3_export import sha
from prepare_c2_recording_manifest import OUTPUT

ROOT = Path(__file__).resolve().parent


def longest_candidate_run(features, times):
    f = features.reshape(-1,8,7)
    enemies = (f[:,:,5] > .5) & (f[:,:,4] >= .1)
    strong = enemies & (f[:,:,4] >= .4) & (f[:,:,2]*f[:,:,3] >= .0003)
    valid = (enemies.sum(axis=1) == 1) & (strong.sum(axis=1) == 1)
    best, current = [], []
    for i in range(len(times)):
        candidate = int(np.argmax(strong[i]))
        contiguous = (current and int(times[i])-int(times[current[-1][0]]) <= 100_000_000
                      and np.linalg.norm(f[i,candidate,:2]-f[current[-1][0],current[-1][1],:2]) <= .12)
        if not valid[i] or not contiguous:
            if len(current) > len(best):
                best = current
            current = []
        if valid[i]:
            current.append((i,candidate))
    if len(current) > len(best):
        best = current
    return best


PAGE = r'''<!doctype html><meta charset="utf-8"><title>C2单目标意图复核</title>
<style>body{font:17px system-ui;background:#111827;color:#e5e7eb;max-width:1100px;margin:24px auto;padding:12px}button,select{font:inherit;padding:8px;margin:5px}canvas{width:100%;background:#000}article{border:1px solid #64748b;margin:25px 0;padding:15px}p{line-height:1.6}.note{color:#fbbf24}footer{position:sticky;bottom:0;background:#1f2937;padding:10px}</style>
<h1>C2：单目标跟随意图复核</h1>
<p>黄色框是C1候选，白色十字是屏幕中心。画面来自原始1080p录像，播放仅使用已缓存的采样帧；下方显示你当时的真实手柄记录，不是AI预测。</p>
<p class="note">仅在整个片段中，候选框确实是敌人、始终为同一人、且你确实在跟随或接近瞄准这个人时，选“可用”。看不清、正在转火/转身找人、目标错误或不能确认意图时，选“不确定”或“不可用”。无需因为RX偶尔与位置偏差方向相反就否决；预判和镜头运动也会造成这种情况。</p>
<p>这是16段以内的小规模诊断复核，不是完整训练集。验证片段仍保留验证身份，不加入训练。</p>
<div id="cards"></div><footer><span id="status"></span><button id="save">保存复核JSON</button></footer>
<script>const data=__DATA__;const state={};const cards=document.getElementById('cards');
function status(){document.getElementById('status').textContent=`已复核 ${Object.keys(state).length}/${data.segments.length}`;}
for(const s of data.segments){const a=document.createElement('article');a.innerHTML=`<h2>${s.key} · ${s.split==='train'?'训练候选':'保留验证'} · 原片 ${s.start_seconds.toFixed(2)}–${s.end_seconds.toFixed(2)}秒</h2><canvas width="1920" height="1080"></canvas><p class="info"></p><button class="prev">上一帧</button><button class="play">慢放</button><button class="next">下一帧</button><select><option value="">尚未复核</option><option value="usable">可用：同一真实敌人，正在跟随</option><option value="uncertain">不确定：看不清或意图不明</option><option value="unusable">不可用：框错、转火或未跟随</option></select>`;cards.appendChild(a);const canvas=a.querySelector('canvas'),ctx=canvas.getContext('2d');let index=0,timer=null;let token=0;
function draw(){const f=s.frames[index],im=new Image(),request=++token;im.onload=()=>{if(request!==token)return;ctx.drawImage(im,0,0);const b=f.box;ctx.strokeStyle='#facc15';ctx.lineWidth=5;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(930,540);ctx.lineTo(990,540);ctx.moveTo(960,510);ctx.lineTo(960,570);ctx.stroke();};im.src=f.image;a.querySelector('.info').textContent=`${index+1}/${s.frames.length} · 原片 ${f.seconds.toFixed(2)}秒 · 原始帧 ${f.frame_id} · 人类 RX ${f.action[0].toFixed(3)} / RY ${f.action[1].toFixed(3)} · RB开火 ${f.action[2]} / LB瞄准 ${f.action[3]} · 候选置信度 ${f.score.toFixed(2)}`;}
function pause(){if(timer){clearInterval(timer);timer=null;}a.querySelector('.play').textContent='慢放';}
a.querySelector('.prev').onclick=()=>{pause();index=(index+s.frames.length-1)%s.frames.length;draw();};a.querySelector('.next').onclick=()=>{pause();index=(index+1)%s.frames.length;draw();};a.querySelector('.play').onclick=()=>{if(timer){pause();return;}a.querySelector('.play').textContent='暂停';timer=setInterval(()=>{index=(index+1)%s.frames.length;draw();},200);};a.querySelector('select').onchange=e=>{if(e.target.value)state[s.key]=e.target.value;else delete state[s.key];status();};draw();}
document.getElementById('save').onclick=()=>{if(Object.keys(state).length!==data.segments.length){alert('请先复核全部片段；无法确认的选“不确定”。');return;}const payload={review_schema:'c2_target_intent_review_v1',candidate_sha256:data.candidate_sha256,decisions:state};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='c2_target_intent_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};status();</script>'''


def main():
    manifest = OUTPUT / 'recording_manifest.json'
    digest = sha(manifest)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    candidates = []
    for row in rows:
        if row['split'] not in ('train','val'):
            continue
        folder = OUTPUT / 'cache' / row['key']
        receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
        provenance = receipt['provenance']
        if (provenance['manifest_sha256'] != digest or provenance['action_schema'] != ACTION_SCHEMA
                or provenance['input_stream'] != 'gamepad.csv' or provenance['key'] != row['key']):
            raise ValueError('Cache provenance differs')
        features = np.load(folder / 'features.npy', mmap_mode='r')
        times = np.load(folder / 'timestamps.npy', mmap_mode='r')
        run = longest_candidate_run(features, times)
        if len(run) >= 8:
            candidates.append((row, run))
    # Keep days represented; never convert held-out validation into training.
    chosen = []
    for day in ('DAY1','DAY2','DAY3','DAY4'):
        group = [item for item in candidates if item[0]['day'] == day and item[0]['split'] == 'train']
        chosen.extend(sorted(group,key=lambda item:(-len(item[1]),item[0]['key']))[:3])
    chosen.extend(sorted([item for item in candidates if item[0]['split'] == 'val'],
                         key=lambda item:(-len(item[1]),item[0]['key']))[:4])
    if not chosen:
        raise ValueError('No candidates; do not loosen gates silently')
    out = ROOT / 'runs/c2_target_review' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    segments = []
    for n, (row, full_run) in enumerate(chosen,1):
        folder = OUTPUT / 'cache' / row['key']
        receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
        for name,value in receipt['cache_files_sha256'].items():
            if sha(folder / name) != value:
                raise ValueError('Cache changed')
        source = Path(row['episode_path'])
        if sha(source / 'video.mkv') != receipt['video_sha256']:
            raise ValueError('Video changed')
        for name,value in row['files_sha256'].items():
            if sha(source / name) != value:
                raise ValueError('Source metadata changed')
        with (source / 'frames.csv').open(newline='',encoding='utf-8-sig') as handle:
            source_start_ns = int(next(csv.DictReader(handle))['source_present_ns'])
        length = min(24,len(full_run))
        begin = (len(full_run)-length)//2
        run = full_run[begin:begin+length]
        features = np.load(folder / 'features.npy', mmap_mode='r').reshape(-1,8,7)
        actions = np.load(folder / 'actions.npy', mmap_mode='r')
        frame_ids = [row['samples'][i]['frame_id'] for i,_ in run]
        selector = '+'.join(f'eq(n,{i})' for i in frame_ids)
        image_folder = out / row['key']
        image_folder.mkdir()
        frames = []
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([resolve_executable('ffmpeg'),'-v','error','-threads','2',
                '-i',str(source/'video.mkv'),'-vf',f"select='{selector}'", '-fps_mode','vfr',
                '-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,stderr=errors)
            try:
                for k,(i,candidate) in enumerate(run):
                    raw = read_exact(process.stdout,1920*1080*3)
                    if len(raw) != 1920*1080*3:
                        raise ValueError('Missing selected original frame')
                    image = np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
                    filename = image_folder / f'{k:03d}.jpg'
                    if not cv2.imwrite(str(filename),image,[cv2.IMWRITE_JPEG_QUALITY,90]):
                        raise ValueError('Image write failed')
                    f = features[i,candidate]
                    sample = row['samples'][i]
                    seconds = (sample['timestamp_ns']-source_start_ns)/1e9
                    frames.append(dict(image=filename.relative_to(out).as_posix(),frame_id=frame_ids[k],
                        cache_index=i,source_present_ns=sample['timestamp_ns'],
                        action_timestamp_ns=sample['action_timestamp_ns'],seconds=seconds,
                        box=[float((f[0]-f[2]/2)*1920),float((f[1]-f[3]/2)*1080),
                             float((f[0]+f[2]/2)*1920),float((f[1]+f[3]/2)*1080)],
                        score=float(f[4]),action=actions[i].tolist()))
                if process.stdout.read(1):
                    raise ValueError('Unexpected additional selected frame')
                process.stdout.close()
                if process.wait(timeout=30):
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors='replace'))
            finally:
                if process.poll() is None:
                    process.kill();process.wait(timeout=10)
        segments.append(dict(key=row['key'],split=row['split'],day=row['day'],
            source_video=str(source/'video.mkv'),start_seconds=frames[0]['seconds'],
            end_seconds=frames[-1]['seconds'],frames=frames))
        print(f'{n}/{len(chosen)} {row["key"]}：{len(frames)}张原始1080p帧',flush=True)
    report = dict(review_schema='c2_target_intent_candidates_v1',manifest_sha256=digest,
        controller_output=False,training_started=False,segments=segments,
        selection='Top8 detector features: exactly one enemy >=0.1, candidate >=0.4, '
                  'area>=0.0003, neighboring normalized center distance<=0.12, gap<=100ms. '
                  'Unverified target identity and human intention; manual acceptance required.')
    candidate = out / 'candidates.json'
    candidate.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['candidate_sha256'] = sha(candidate)
    payload = json.dumps(report,ensure_ascii=False).replace('<','\\u003c')
    (out / 'review.html').write_text(PAGE.replace('__DATA__',payload),encoding='utf-8')
    print('候选分组：',dict(Counter(s['split'] for s in segments)))
    print('复核页：',out / 'review.html')


if __name__ == '__main__':
    main()
