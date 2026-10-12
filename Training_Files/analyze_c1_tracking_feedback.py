"""Read-only log analysis plus report export; no model, services, or video run."""
import argparse
import json
from pathlib import Path

from c1_tracking import overlap

ROOT=Path(__file__).resolve().parent
FEEDBACK=[('006',1.,3.,['A4','F1','D4']),('006',3.7,4.3,['B1']),
          ('008',.7,1.3,['C3','A4']),('009',1.5,4.,['F1','A4','D2']),
          ('021',None,None,['D2']),('035',None,None,['A4','B1','F1','F2','F3','E3','D2'])]


def analyze(run):
    windows=[]
    for clip,start,end,codes in FEEDBACK:
        path=run/f'episode_{int(clip):06d}'/'tracks.jsonl'
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        origin=rows[0]['source_present_ns']
        chosen=[r for r in rows if not r['reused_timestamp'] and
                (start is None or start<=(r['source_present_ns']-origin)/1e9<=end)]
        born=set();confirmed=set();duplicate_frames=0
        for r in chosen:
            born.update(tr['track_id'] for tr in r['tracks'] if tr['track_id'] is not None and tr['hits']==1 and tr['position_valid'])
            confirmed.update(tr['track_id'] for tr in r['tracks'] if tr.get('confirmed') and tr['position_valid'])
            boxes=[d['box'] for d in r['detections']]
            duplicate=False
            for i,a in enumerate(boxes):
                for b in boxes[i+1:]:
                    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
                    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                    if overlap(a,b)>.6 or intersection/max(1,min(area_a,area_b))>.85:duplicate=True
            duplicate_frames+=duplicate
        windows.append(dict(clip=clip,start_s=start,end_s=end,user_codes=codes,unique_frames=len(chosen),
            raw_zero_detection_frames=sum(not r['detections'] for r in chosen),
            detection_present_no_confirmed_observation_frames=sum(bool(r['detections']) and not any(
                tr['position_valid'] and tr.get('confirmed') for tr in r['tracks']) for r in chosen),
            camera_rejected_frames=sum(r.get('association_diagnostics',{}).get('camera',{}).get('status')=='rejected' for r in chosen),
            born_tracks_in_window=len(born),observed_confirmed_ids=len(confirmed),
            geometrically_overlapping_detection_frames=duplicate_frames))
    return windows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    windows=analyze(args.run)
    report=ROOT/'diagnostics/c1_tracking_v3_feedback';report.mkdir(parents=True,exist_ok=True)
    payload=dict(source_run=str(args.run),time_semantics='native source_present_ns elapsed from first frame',
        point_window_note='1s/4s reports expanded by ±0.3s for diagnosis, not asserted exact error boundaries',
        caveat='No person GT: zero-detection frame may be legitimate; overlap may be two people. Counts are diagnostic signals, not accuracy.',windows=windows)
    (report/'feedback.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# V3 人工反馈与日志核对','',f'来源：`{args.run}`','',
           '时间区间使用原生呈现时间；1s/4s点位扩展±0.3s。以下信号不能自动证明漏检、重复框或串号。','',
           '|片段/时间|人工代码|帧数|原始检测为空|有检测但无已确认当前框|镜头拒绝|新轨迹|几何重叠帧|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for w in windows:
        timing='全片' if w['start_s'] is None else f"{w['start_s']}–{w['end_s']}s"
        lines.append(f"|{w['clip']} {timing}|{'+'.join(w['user_codes'])}|{w['unique_frames']}|{w['raw_zero_detection_frames']}|{w['detection_present_no_confirmed_observation_frames']}|{w['camera_rejected_frames']}|{w['born_tracks_in_window']}|{w['geometrically_overlapping_detection_frames']}|")
    lines+=['','已确认的用户反馈作为定性验收记录，不直接修改训练/验证标签。',
            'F类优先修生命周期与关联竞争；A4/B1/C3/D/E类需原始框/图像证据，不能用隐藏未确认框或预测代替检测改进。']
    (report/'反馈核对.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2));print(report/'反馈核对.md')


if __name__=='__main__':main()
