"""Bounded extraction of user-reported errors; no inference or label changes."""
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from fps_recorder.utils import resolve_executable

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/c1_tracking/20261010_014322_036110'
OUT=RUN/'reported_false_positives';OUT.mkdir(exist_ok=True)
reports=[]
for episode,seconds in [(82,5),(82,5.5),(320,1),(320,1.4),(320,1.6)]:
    name=f'episode_{episode:06d}'
    source=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY3')/name
    meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    fps=meta['encoder']['output_fps']
    rows=[json.loads(s) for s in (RUN/name/'tracks.jsonl').read_text(encoding='utf-8').splitlines()]
    # User times refer to playback. Frame index selects a decoded video frame;
    # native timestamp is preserved from the same frame's log, no action join.
    index=min(round(seconds*fps),len(rows)-1)
    row=rows[index]
    path=OUT/f'{name}_{seconds}s.png'
    subprocess.run([resolve_executable('ffmpeg'),'-v','error','-i',str(source/'video.mkv'),
                    '-vf',f'select=eq(n\\,{index})','-frames:v','1','-y',str(path)],check=True,timeout=30)
    image=Image.open(path).convert('RGB');draw=ImageDraw.Draw(image)
    for j,d in enumerate(row['detections']):
        draw.rectangle(d['box'],outline='red' if d['class']==0 else 'cyan',width=3)
        draw.text((d['box'][0],max(0,d['box'][1]-14)),f"d{j} class={d['class']} conf={d['score']:.3f}",fill='yellow')
    image.save(OUT/f'{name}_{seconds}s_detections.png')
    window=rows[max(0,index-round(.5*fps)):min(len(rows),index+round(.5*fps)+1)]
    reports.append(dict(episode=episode,playback_seconds=seconds,frame_index=index,
                        source_present_ns=row['source_present_ns'],frame=row,window=window))
    print(json.dumps(dict(episode=episode,index=index,detections=row['detections'],
                         selected=row['target_selection']),ensure_ascii=False))
(OUT/'evidence.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
