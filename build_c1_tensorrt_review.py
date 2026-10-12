"""Mechanical visualization of existing logs; no model inference or validation run."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from demo_c1_video_tracking import ROOT, load_clip


def read_records(path):
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {r['frame_id']: r for r in rows}
    if len(result) != len(rows): raise ValueError('重复日志帧编号')
    return result


def annotated(image, detections, heading):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 25)
    # Reserve an added header instead of obscuring scene/HUD pixels.
    panel = Image.new('RGB', (image.width, image.height+70), '#141b24')
    panel.paste(image, (0, 70))
    draw = ImageDraw.Draw(panel)
    draw.text((20, 16), heading+'  · 红=敌人，蓝=队友', font=font, fill='white')
    for i, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = detection['box']
        color = '#ff5757' if detection['class'] == 0 else '#46baff'
        label = f"框{i} {'敌人' if detection['class'] == 0 else '队友'} {detection['score']:.4f}"
        draw.rectangle((x1, y1+70, x2, y2+70), outline=color, width=3)
        x = max(0, min(image.width-360, x1)); y = max(70, min(image.height+35, y1+40))
        draw.text((x, y), label, font=font, fill=color, stroke_width=2, stroke_fill='black')
    return panel


def build(run):
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    changed = json.loads((run / 'changed_frames.json').read_text(encoding='utf-8'))
    source = Path(summary['source'])
    _, frames = load_clip(source)
    pt = read_records(run / 'pt_frames.jsonl'); engine = read_records(run / 'engine_frames.jsonl')
    changed = sorted(changed, key=lambda r: r['frame_id'])
    for row in changed:
        fid = row['frame_id']; stamp = row['source_present_ns']
        if not (pt[fid]['source_present_ns'] == engine[fid]['source_present_ns']
                == int(frames[fid]['source_present_ns']) == stamp):
            raise ValueError('源帧与两个日志的原生时间戳不一致')
    output = run / ('差异画面_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    wanted = {row['frame_id']: (i, row) for i, row in enumerate(changed, 1)}
    cap = cv2.VideoCapture(str(source / 'video.mkv'))
    if not cap.isOpened(): raise RuntimeError('无法打开原视频')
    entries = []; origin = int(frames[0]['source_present_ns'])
    try:
        for fid in range(max(wanted, default=-1)+1):
            ok, bgr = cap.read()
            if not ok: raise ValueError(f'解码提前结束：{fid}')
            if fid not in wanted: continue
            number, difference = wanted[fid]
            seconds = (difference['source_present_ns']-origin)/1e9
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            label = f'{number:02d} · {seconds:.3f}s · 源帧{fid}'
            image.save(output / f'{number:02d}_原图.jpg', quality=98, subsampling=0)
            panels = []
            for backend, title in [('pt', '原模型 PT'), ('engine', 'TensorRT FP16')]:
                panel = annotated(image, (pt if backend == 'pt' else engine)[fid]['detections'], label+' · '+title)
                panel.save(output / f'{number:02d}_{backend}.jpg', quality=98, subsampling=0)
                panels.append(panel)
            pair = Image.new('RGB', (panels[0].width*2, panels[0].height))
            pair.paste(panels[0], (0, 0)); pair.paste(panels[1], (panels[0].width, 0))
            pair.save(output / f'{number:02d}_对比.jpg', quality=98, subsampling=0)
            reasons = []
            if difference['spatial_class_disagreements']: reasons.append('同位置类别不同（优先检查）')
            if difference['pt_unmatched'] or difference['engine_unmatched']:
                reasons.append(f"未匹配：PT {difference['pt_unmatched']} / TRT {difference['engine_unmatched']}")
            if any(v < .9 for v in difference['matched_ious']): reasons.append('框形状变化')
            entries.append(dict(number=f'{number:02d}', frame_id=fid, seconds=seconds,
                                reasons=reasons, pt=pt[fid]['detections'], engine=engine[fid]['detections']))
    finally: cap.release()
    (output / 'review_manifest.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = []
    for entry in entries:
        n = entry['number']; reason = html.escape('；'.join(entry['reasons']))
        cards.append(f'''<section><h2>{n} · {entry['seconds']:.3f}s · {reason}</h2>
<p><a href="{n}_原图.jpg" target="_blank">原图</a> · <a href="{n}_pt.jpg" target="_blank">原模型高清</a> ·
<a href="{n}_engine.jpg" target="_blank">TensorRT高清</a> · <a href="{n}_对比.jpg" target="_blank">全尺寸对比</a></p>
<div class="pair"><a href="{n}_pt.jpg" target="_blank"><img loading="lazy" src="{n}_pt.jpg" alt="{n}原模型"></a>
<a href="{n}_engine.jpg" target="_blank"><img loading="lazy" src="{n}_engine.jpg" alt="{n}TensorRT"></a></div></section>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TensorRT差异画面</title>
<style>body{background:#131921;color:#e9edf2;font:16px 'Microsoft YaHei',sans-serif;margin:24px}a{color:#85c8ff}
section{border-top:1px solid #526173;padding:18px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;height:auto}
h2{font-size:19px}p{line-height:1.8}</style><h1>TensorRT差异画面 · 左原模型，右TensorRT</h1>
<p>仅绘制已有检测日志，没有重新推理。点击图片查看原尺寸。框1/框2只是本图检测序号，不是人物跟踪编号。
未匹配或框形状变化不自动等于误检/漏检。反馈图片编号＋哪侧有问题＋问题类型即可。</p>'''
    (output / '差异浏览.html').write_text(document+'\n'.join(cards)+'</html>', encoding='utf-8')
    print(f'已生成 {len(entries)} 张高清差异对比；未加载检测或外观模型。\n浏览文件：{output / "差异浏览.html"}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    build(parser.parse_args().run.resolve())
