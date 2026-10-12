"""Bounded integrity/provenance audit of new user-selected videos; no inference."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from prepare_c1_day2_preview import audit_episode
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY4')
EPISODES=[1,3,6,7,9]
OUTPUT=ROOT/'diagnostics/c1_day4_unseen_validation_v1'
TRAIN=ROOT/'c1_yolo_dataset/day4_joint_errors_v1/source_manifest.json'


def main():
    rows=json.loads(TRAIN.read_text(encoding='utf-8'))
    source_paths={str(Path(r['episode_path']).resolve()).lower() for r in rows if r.get('episode_path')}
    metadata_hashes={r['source_metadata_sha256'] for r in rows if r.get('source_metadata_sha256')}
    frames_hashes={r['source_frames_sha256'] for r in rows if r.get('source_frames_sha256')}
    paths=[SOURCE/f'episode_{n:06d}' for n in EPISODES]
    with ThreadPoolExecutor(max_workers=3) as pool:audits=list(pool.map(audit_episode,paths))
    for audit,path in zip(audits,paths,strict=True):
        if audit['audit_status']=='failed':raise ValueError(f'{path.name}: {audit.get("error")}')
        if (str(path.resolve()).lower() in source_paths or audit['source_metadata_sha256'] in metadata_hashes
                or audit['source_frames_sha256'] in frames_hashes):
            raise ValueError('Selected clip source matches existing dataset; not unseen')
        with (path/'frames.csv').open(newline='') as handle:frames=list(csv.DictReader(handle))
        if [int(f['frame_id']) for f in frames]!=list(range(len(frames))):raise ValueError('Frame/video index contract differs')
        audit['source_video_sha256']=sha(path/'video.mkv')
        print(f'{path.name}: {audit["frame_count"]}帧，完整解码通过；提示={audit["warnings"]}',flush=True)
    report=dict(stage='Stage 1 — Local Combat Policy',source=str(SOURCE),episodes=EPISODES,
        train_manifest_sha256=sha(TRAIN),clips=audits,training_source_match=False,
        status='new user-selected visual validation clips; not added to training or annotation',
        scope='Source paths/metadata/frame CSV differ from dataset. Does not certify different matches or all-frame perceptual uniqueness.',
        no_action_alignment=True,model_inference_started=False)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'selection_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'共{sum(a["frame_count"] for a in audits)}帧；清单：{OUTPUT}',flush=True)


if __name__=='__main__':main()
