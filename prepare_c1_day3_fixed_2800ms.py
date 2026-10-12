"""Mechanical Day3 2.8-second sampling and joint-n prelabels; no training/service."""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import prepare_c1_day3_recordings as shared

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY3')
OUTPUT=ROOT/'c1_annotations/day3_fixed_2800ms_v1'
WEIGHTS=ROOT/'runs/c1_yolo/day3_yolo26n_1280_joint_pretrained_v1/weights/best.pt'
EXPECTED_SHA='12c41173c4b20bc8a62ea92faaeb9295c50d43c87a149ca7c176a085337429a2'


def choose_frame(frames):
    if not frames:raise ValueError('Empty frames')
    start=int(frames[0]['source_present_ns']);target=start+2_800_000_000
    if int(frames[-1]['source_present_ns'])<target:
        return None
    unique=[f for f in frames if not int(f.get('is_duplicate',0))]
    if not unique:raise ValueError('No unique source frames')
    selected=min(unique,key=lambda f:(abs(int(f['source_present_ns'])-target),int(f['source_present_ns'])))
    if abs(int(selected['source_present_ns'])-target)>100_000_000:
        raise ValueError('No unique frame within 100ms of fixed target')
    return selected


def extract(audit):
    path=Path(audit['episode_path'])
    if audit['audit_status']=='failed':raise ValueError(f'Failed episode: {path}')
    if shared.sha(path/'metadata.json')!=audit['source_metadata_sha256'] or shared.sha(path/'frames.csv')!=audit['source_frames_sha256']:
        raise ValueError('Previously audited source changed')
    with (path/'frames.csv').open(newline='') as handle:frames=list(csv.DictReader(handle))
    frame=choose_frame(frames)
    if frame is None:return dict(episode=path.name,skipped='shorter than 2.8s; no substitute frame')
    index=int(frame['frame_id'])
    image=OUTPUT/'images'/f'day3_t2800_{path.name}_f{index:06d}.png'
    if not image.exists():
        subprocess.run([shared.resolve_executable('ffmpeg'),'-v','error','-threads','1','-i',str(path/'video.mkv'),
                        '-vf',f'select=eq(n\\,{index})','-fps_mode','vfr','-frames:v','1','-threads','1','-n',str(image)],
                       capture_output=True,check=True,timeout=180)
    with Image.open(image) as im:
        im.load();width,height=im.size
        if (width,height)!=(1920,1080):raise ValueError('Unexpected frame dimensions')
        pixel_sha=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
    return dict(episode=path.name,episode_path=str(path),frame_id=index,
                frame_timestamp_ns=int(frame['source_present_ns']),image_path=str(image),image_name=image.name,
                width=width,height=height,image_sha256=shared.sha(image),pixel_sha256=pixel_sha,
                requested_offset_ns=2_800_000_000,actual_offset_ns=int(frame['source_present_ns'])-int(frames[0]['source_present_ns']),
                split='unassigned',match_group='unconfirmed',qpc_session=audit['qpc_session'],
                audit_status=audit['audit_status'],warnings=audit['warnings'],
                source_metadata_sha256=audit['source_metadata_sha256'],source_frames_sha256=audit['source_frames_sha256'])


def main():
    if shared.sha(WEIGHTS)!=EXPECTED_SHA:raise ValueError('Frozen prelabel model changed')
    audits=json.loads((ROOT/'c1_annotations/day3_v4_prelabel_v1/recording_audit.json').read_text(encoding='utf-8'))
    if {p.name for p in SOURCE.glob('episode_*') if p.is_dir()}!={a['episode'] for a in audits}:
        raise ValueError('Source roster differs from previous full decode audit')
    OUTPUT.mkdir(exist_ok=True)
    for name in ('images','contacts'):(OUTPUT/name).mkdir(exist_ok=True)
    records=[];skipped=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(extract,sorted(audits,key=lambda a:a['episode'])),1):
            (skipped if 'skipped' in r else records).append(r)
            if i%20==0:print(f'固定2.8秒抽帧 {i}/{len(audits)}',flush=True)
    records=[dict(r,preview_id=i) for i,r in enumerate(records,1)]
    previous=json.loads((ROOT/'c1_annotations/day3_v4_prelabel_v1/manifest.json').read_text(encoding='utf-8'))
    old_pixels={r['pixel_sha256']:r['image_name'] for r in previous};seen={};duplicates=[]
    for r in records:
        if r['pixel_sha256'] in old_pixels or r['pixel_sha256'] in seen:
            duplicates.append(dict(image=r['image_name'],same_pixels_as=old_pixels.get(r['pixel_sha256'],seen.get(r['pixel_sha256']))))
        seen[r['pixel_sha256']]=r['image_name']
    manifest=OUTPUT/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=records:raise ValueError('Existing manifest differs')
    shared.save(manifest,records);shared.save(OUTPUT/'exact_duplicates.json',duplicates)
    shared.save(OUTPUT/'preparation_summary.json',dict(episodes=len(audits),images=len(records),skipped=skipped,
        exact_duplicates=len(duplicates),sampling='nearest unique native source timestamp at first frame +2.8 seconds; max deviation 100ms',
        prior_complete_decode_audit_reused=True,action_alignment_performed=False,training_started=False,
        split='all usable new frames intended for train; original validation preserved; unknown matches not randomly split',
        note='Every eligible episode retained for review; exact duplicates must be reconciled before training merge'))
    shared.OUTPUT=OUTPUT;shared.WEIGHTS=WEIGHTS
    shared.MODEL_VERSION='joint_n_fixed2800_conf005';shared.PRELABEL_NAME='联合n'
    shared.contacts(records);shared.prelabel()


if __name__=='__main__':main()
