"""Mechanical cached-log review, no inference, no tracker replay, no services."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from build_c1_tensorrt_review import annotated, read_records
from demo_c1_video_tracking import load_clip, render


def build(clip):
    summary=json.loads((clip/'summary.json').read_text(encoding='utf-8'))
    source=Path(summary['source']); _,frames=load_clip(source)
    rows=read_records(clip/'tracks.jsonl')
    start=int(frames[0]['source_present_ns']); end=int(frames[-1]['source_present_ns'])
    selected=[]; previous=-1e30
    for row in rows.values():
        timestamp=row['source_present_ns']
        if row.get('reused_timestamp') or timestamp<start+(end-start)/2: continue
        if not any(t['state']=='unassociated' for t in row['tracks']): continue
        seconds=(timestamp-start)/1e9
        if seconds-previous<.35: continue
        selected.append(row); previous=seconds
        if len(selected)>=12: break
    if not selected: raise ValueError('后半程没有可整理的未关联帧')
    output=clip/('未关联核查_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));output.mkdir()
    wanted={r['frame_id']:(i,r) for i,r in enumerate(selected,1)}
    cap=cv2.VideoCapture(str(source/'video.mkv')); entries=[];cards=[]
    if not cap.isOpened():raise RuntimeError('无法打开原片段')
    try:
        for fid in range(max(wanted)+1):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'提前解码结束：{fid}')
            if fid not in wanted:continue
            number,row=wanted[fid];seconds=(row['source_present_ns']-start)/1e9
            if row['source_present_ns']!=int(frames[fid]['source_present_ns']):raise ValueError('日志与源帧时间不一致')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);original=Image.fromarray(rgb)
            heading=f'{number:02d} · {seconds:.3f}s · 源帧{fid}'
            original.save(output/f'{number:02d}_原图.jpg',quality=98,subsampling=0)
            detection=annotated(original,row['detections'],heading+' · 原始检测')
            tracked=render(rgb,row['detections'],row['tracks'],'',show_unassociated=True)
            panel=Image.new('RGB',(1920,1150),'#141b24');panel.paste(tracked,(0,70))
            ImageDraw.Draw(panel).text((20,16),heading+' · 跟踪结果（灰=未关联/待确认）',
                font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25),fill='white')
            detection.save(output/f'{number:02d}_检测.jpg',quality=98,subsampling=0)
            panel.save(output/f'{number:02d}_跟踪.jpg',quality=98,subsampling=0)
            entries.append(dict(number=f'{number:02d}',frame_id=fid,source_present_ns=row['source_present_ns'],
                                seconds=seconds,detections=row['detections'],tracks=row['tracks'],
                                diagnostics=row['association_diagnostics']))
            cards.append(f'''<section><h2>{number:02d} · {seconds:.3f}s</h2><p>
<a href="{number:02d}_原图.jpg" target="_blank">原图</a> · 点击两侧图片看原尺寸</p>
<div><a href="{number:02d}_检测.jpg" target="_blank"><img src="{number:02d}_检测.jpg" loading="lazy"></a>
<a href="{number:02d}_跟踪.jpg" target="_blank"><img src="{number:02d}_跟踪.jpg" loading="lazy"></a></div></section>''')
    finally:cap.release()
    (output/'review_manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>35后半程未关联核查</title>
<style>body{background:#131921;color:white;font:16px 'Microsoft YaHei';margin:24px}a{color:#85c8ff}
section{border-top:1px solid #526173;padding:16px 0}section div{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}</style>
<h1>35后半程 · 左原始检测，右跟踪</h1><p>使用已有日志绘制，没有重新推理。
选取后半程出现未关联的代表帧（间隔至少0.35秒，最多12张），不是全部失败帧。
请指出重叠框是同一个人还是不同人、实际敌我类别、灰框是否仍正确覆盖人物。
检测的框序号不是人物跟踪ID。点击原图帮助判断。</p>'''
    (output/'未关联浏览.html').write_text(page+'\n'.join(cards)+'</html>',encoding='utf-8')
    print(f'完成：{len(entries)}帧，{output / "未关联浏览.html"}')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--clip',type=Path,required=True)
    build(parser.parse_args().clip.resolve())
