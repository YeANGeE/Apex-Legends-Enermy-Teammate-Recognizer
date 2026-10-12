"""Merge manually corrected DAY4 supplement; preserve held-out clip sources."""
import json
from pathlib import Path
import prepare_c1_day3_fixed_2800ms_training as shared

if __name__ == '__main__':
    shared.BASE = shared.ROOT/'c1_yolo_dataset/day4_joint_errors_v1'
    shared.NEW = shared.ROOT/'c1_yolo_dataset/day4_supplement_normalized_v1'
    shared.OUTPUT = shared.ROOT/'c1_yolo_dataset/day4_joint_supplement_v1'
    shared.EXPECTED_BASE_COUNTS = {'train':1731, 'val':62}
    shared.ANNOTATION_SOURCE = 'day4_E2_E4_E10_E11_E12_completed_manual_annotation'
    shared.NEW_ENCOUNTER_ID = 'day4_match_boundaries_unconfirmed_all_train'
    shared.NEW_SOURCE_DOMAIN = 'day4_real_match'
    shared.NEW_IMAGE_PREFIX = 'day4_supplement_'
    if shared.sha(shared.BASE/'source_manifest.json') != 'f4e958ffe913040e236fed4fa58fc2e9468c73e11ebe0f9e38c6990737c19a25':
        raise ValueError('Previous dataset changed')
    held = json.loads((shared.ROOT/'diagnostics/c1_day4_unseen_validation_v1/selection_manifest.json').read_text(encoding='utf-8'))
    held_paths = {str(Path(a['episode_path']).resolve()).lower() for a in held['clips']}
    held_videos = {a['source_video_sha256'] for a in held['clips']}
    for folder in (shared.BASE, shared.NEW):
        for r in json.loads((folder/'source_manifest.json').read_text(encoding='utf-8')):
            if (r.get('episode_path') and str(Path(r['episode_path']).resolve()).lower() in held_paths) or r.get('source_video_sha256') in held_videos:
                raise ValueError('Held-out validation clip found in dataset')
    shared.main()
    path = shared.OUTPUT/'dataset_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary.update(held_out_day4_episodes=[1,3,6,7,9], held_out_sources_used=False,
        warning='Episode source separation verified; match independence and near-duplicate visual content not certified.')
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
