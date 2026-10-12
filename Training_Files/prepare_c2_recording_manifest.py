"""Preliminary all-day audit and causal action selection; no model inference."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from c2_policy import backward_index,action_values,ACTION_ORDER,ACTION_SCHEMA
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/FPS_AI_Dataset')
OUTPUT=ROOT/'c2_dataset/day1234_bc_rb_lb_v2'
SOURCES={'DAY1':'Apex_Day1','DAY2':'Apex_YOLO_DAY2','DAY3':'Apex_YOLO_DAY3','DAY4':'Apex_YOLO_DAY4',
         'DAY2_VAL':'Apex_YOLO_DAY2_VAL'}


def main():
    with (ROOT/'c1_day1_formal_manifest.csv').open(encoding='utf-8-sig',newline='') as h:
        groups={r['clip_id']:(r['split'],r['encounter_id']) for r in csv.DictReader(h)}
    rows=[];excluded=[]
    for day,folder in SOURCES.items():
        for path in sorted((SOURCE/folder).glob('episode_*')):
            try:
                meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
                if meta.get('recording_status')!='complete' or meta.get('encoder_queue_overflow') or meta.get('encoder_errors'):
                    raise ValueError('Incomplete/overflow/encoding errors')
                with (path/'frames.csv').open(newline='') as h:frames=list(csv.DictReader(h))
                with (path/'gamepad.csv').open(newline='') as h:actions=list(csv.DictReader(h))
                if len(frames)!=meta['frame_count'] or [int(f['frame_id']) for f in frames]!=list(range(len(frames))):
                    raise ValueError('Frame roster mismatch')
                times=[int(a['timestamp_ns']) for a in actions]
                ftimes=[int(f['source_present_ns']) for f in frames]
                if not times or any(b<a for a,b in zip(times,times[1:])) or any(b<a for a,b in zip(ftimes,ftimes[1:])):
                    raise ValueError('Invalid native timeline')
                split='train';group=day+':unknown_match'
                if day=='DAY1' and path.name in groups:
                    split,g=groups[path.name];group='DAY1:'+g
                if day=='DAY2_VAL':split='diagnostic'
                ep=int(path.name.split('_')[-1])
                if (day=='DAY3' and ep in [43,47,79,82,150,178,320]) or (day=='DAY4' and ep in [1,3,6,7,9]):
                    split='diagnostic'
                samples=[];next_t=ftimes[0];missing=stale=0
                for f,t in zip(frames,ftimes):
                    if int(f['is_duplicate']) or t<next_t:continue
                    next_t=t+50_000_000
                    j=backward_index(times,t)
                    if j<0:missing+=1;continue
                    if t-times[j]>100_000_000:stale+=1;continue
                    a=actions[j];values=action_values(a)
                    samples.append(dict(frame_id=int(f['frame_id']),timestamp_ns=t,action_timestamp_ns=times[j],action=values))
                if len(samples)<10:raise ValueError('Too few valid action samples')
                rows.append(dict(key=day+'_'+path.name,day=day,episode_path=str(path),split=split,group=group,
                    samples=samples,frames=len(frames),missing_predecessor=missing,stale_action_samples=stale,
                    fire_samples=sum(s['action'][2]>.1 for s in samples),
                    files_sha256={name:sha(path/name) for name in ['metadata.json','frames.csv','gamepad.csv']},
                    video_bytes=(path/'video.mkv').stat().st_size))
            except (ValueError,KeyError,OSError) as exc:
                excluded.append(dict(path=str(path),reason=str(exc)))
        print(day,':',sum(r['day']==day for r in rows),'段完成初步对齐',flush=True)
    # Formal grouped validation is DAY1's existing encounter split only.
    if not any(r['split']=='val' for r in rows):raise ValueError('No grouped validation recordings')
    train_groups={r['group'] for r in rows if r['split']=='train'}
    if train_groups & {r['group'] for r in rows if r['split']=='val'}:raise ValueError('Train/val group overlap')
    candidates=[r for r in rows if r['split']=='train' and r['fire_samples']>=10
                and len(r['samples'])-r['fire_samples']>=5 and 20<=len(r['samples'])<=400]
    if not candidates:raise ValueError('No short firing clip for initial pipeline test')
    smoke=max(candidates,key=lambda r:r['fire_samples']/len(r['samples']))['key']
    summary=dict(stage='Stage 1 — Local Combat Policy',input_stream='gamepad.csv',
        action_order=ACTION_ORDER,action_schema=ACTION_SCHEMA,button_mapping={'fire':'rb','aim':'lb'},
        alignment='latest timestamp_ns <= frame source_present_ns',sampling='native source time >=50ms since previous selected sample',
        clips_by_split=dict(Counter(r['split'] for r in rows)),clips_by_day=dict(Counter(r['day'] for r in rows)),
        action_samples_by_split={s:sum(len(r['samples']) for r in rows if r['split']==s) for s in ['train','val','test','diagnostic']},
        smoke_clip=smoke,excluded=excluded,training_started=False,controller_output=False,
        scope='Metadata/action audit only; complete decode verified during feature-cache preparation. DAY2-4 match boundaries unknown; no random frame split.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=OUTPUT/'recording_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=rows:raise ValueError('Existing manifest differs')
    manifest.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    (OUTPUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='excluded'},ensure_ascii=False,indent=2))
    print('Excluded:',len(excluded))


if __name__=='__main__':main()
