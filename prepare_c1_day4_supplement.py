"""Bounded half-second sampling and draft annotations of selected DAY4 clips."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import prepare_c1_day3_recordings as shared
from demo_c1_day4_tracking import WEIGHTS, EXPECTED_SHA

ROOT = Path(__file__).resolve().parent
SOURCE = Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY4')
OUTPUT = ROOT/'c1_annotations/day4_supplement_v1'
EPISODES = [2, 4, 10, 11, 12]


def sampled_frames(path, audit):
    with (path/'frames.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if [int(r['frame_id']) for r in rows] != list(range(len(rows))):
        raise ValueError('Noncontiguous video frame IDs')
    available = [r for r in rows if not int(r['is_duplicate'])]
    times = [int(r['source_present_ns']) for r in available]
    if not times or any(b < a for a,b in zip(times,times[1:])):
        raise ValueError('Invalid native timestamps')
    targets = list(range(times[0], times[-1]+1, 500_000_000))
    if times[-1]-targets[-1] >= 250_000_000:
        targets.append(times[-1])
    selected = {}
    for target in targets:
        r = min(available, key=lambda r: abs(int(r['source_present_ns'])-target))
        if abs(int(r['source_present_ns'])-target) > 100_000_000:
            continue
        selected[int(r['frame_id'])] = r
    return [selected[k] for k in sorted(selected)]


def main():
    if shared.sha(WEIGHTS) != EXPECTED_SHA:
        raise ValueError('Day4 model changed')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for directory in ('images','contacts'):
        (OUTPUT/directory).mkdir(exist_ok=True)
    paths = [SOURCE/f'episode_{n:06d}' for n in EPISODES]
    with ThreadPoolExecutor(max_workers=3) as pool:
        audits = list(pool.map(shared.audit_episode, paths))
    held = json.loads((ROOT/'diagnostics/c1_day4_unseen_validation_v1/selection_manifest.json').read_text(encoding='utf-8'))
    held_videos = {a['source_video_sha256'] for a in held['clips']}
    records = []
    shared.OUTPUT = OUTPUT
    shared.selected_frames = sampled_frames
    for audit, path in zip(audits, paths, strict=True):
        if audit['audit_status'] == 'failed':
            raise ValueError(f'{path.name}: failed recording audit')
        audit['source_video_sha256'] = shared.sha(path/'video.mkv')
        if audit['source_video_sha256'] in held_videos:
            raise ValueError('Selected video duplicates a held-out validation clip')
        batch = shared.extract_episode(audit)
        for r in batch:
            r['source_video_sha256'] = audit['source_video_sha256']
        records.extend(batch)
        print(f'{path.name}: {audit["frame_count"]}帧，抽取{len(batch)}张；提示={audit["warnings"]}', flush=True)
    train = json.loads((ROOT/'c1_yolo_dataset/day4_joint_errors_v1/source_manifest.json').read_text(encoding='utf-8'))
    old_pixels = {r['pixel_sha256'] for r in train if r.get('pixel_sha256')}
    kept, duplicates, seen = [], [], set()
    for r in records:
        pixel = r['pixel_sha256']
        if pixel in seen or pixel in old_pixels:
            duplicates.append(dict(image=r['image_name'], reason='within_batch_or_existing_train'))
            continue
        seen.add(pixel)
        r['preview_id'] = len(kept)+1
        kept.append(r)
    manifest = OUTPUT/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != kept:
        raise ValueError('Existing batch differs; do not overwrite')
    shared.save(OUTPUT/'recording_audit.json', audits)
    shared.save(manifest, kept)
    shared.save(OUTPUT/'exact_duplicates.json', duplicates)
    shared.save(OUTPUT/'preparation_summary.json', dict(episodes=EPISODES, images=len(kept),
        sampling='Native source_present_ns every 500ms, nearest nonduplicate within 100ms; no prediction-based filtering',
        held_out_episodes=[1,3,6,7,9], held_out_video_duplicate=False,
        match_independence='Unknown; different episode numbers do not prove different matches',
        discarded_duplicate_images=len(duplicates), training_started=False, action_alignment_performed=False))
    shared.contacts(kept)
    shared.WEIGHTS = WEIGHTS
    shared.MODEL_VERSION = 'day4_supplement_n_conf010_v1'
    shared.PRELABEL_NAME = 'DAY4联合n'
    shared.PRELABEL_CONFIDENCE = .10
    shared.PREVIEW_PREFIX = 'DAY4补样'
    shared.prelabel()


if __name__ == '__main__':
    main()
