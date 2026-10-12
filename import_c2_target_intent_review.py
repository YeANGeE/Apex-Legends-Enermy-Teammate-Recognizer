"""Import explicit segment decisions; preserve validation and source labels."""
import argparse
import json
from collections import Counter
from pathlib import Path

from normalize_c1_day3_export import sha
from prepare_c2_recording_manifest import OUTPUT

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / 'runs/c2_target_review/20261010_231628_310735/candidates.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review',type=Path,default=Path('C:/Users/15207/Downloads/c2_target_intent_review.json'))
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding='utf-8-sig'))
    candidate_digest = sha(CANDIDATES)
    if review.get('review_schema') != 'c2_target_intent_review_v1' or review.get('candidate_sha256') != candidate_digest:
        raise ValueError('Review schema/candidate SHA differs')
    candidate = json.loads(CANDIDATES.read_text(encoding='utf-8'))
    if candidate['manifest_sha256'] != sha(OUTPUT/'recording_manifest.json'):
        raise ValueError('Original manifest changed')
    decisions = review['decisions']
    if set(decisions) != {s['key'] for s in candidate['segments']}:
        raise ValueError('Missing or unexpected review decisions')
    if any(value not in ('usable','uncertain','unusable') for value in decisions.values()):
        raise ValueError('Unsupported decision')
    accepted = []
    for segment in candidate['segments']:
        if decisions[segment['key']] != 'usable':
            continue
        if segment['split'] not in ('train','val'):
            raise ValueError('Unexpected split')
        for frame in segment['frames']:
            if frame['action_timestamp_ns'] > frame['source_present_ns']:
                raise ValueError('Noncausal original action label')
        accepted.append(segment)
    report = dict(review_sha256=sha(args.review),candidate_sha256=candidate_digest,
        manifest_sha256=candidate['manifest_sha256'],decisions=decisions,accepted_segments=accepted,
        segment_counts=dict(Counter(s['split'] for s in accepted)),
        frame_counts={split:sum(len(s['frames']) for s in accepted if s['split']==split) for split in ('train','val')},
        training_started=False,controller_output=False,
        note='Manual target intent accepted only for these displayed segment ranges; no RL rewards, '
             'no new action labels, no automatic expansion to entire episodes.')
    out = ROOT / 'diagnostics/c2_target_intent'
    out.mkdir(parents=True,exist_ok=True)
    target = out / f'accepted_{report["review_sha256"][:12]}.json'
    serialized = json.dumps(report,ensure_ascii=False,indent=2)
    if target.exists() and target.read_text(encoding='utf-8') != serialized:
        raise ValueError('Existing accepted review differs')
    target.write_text(serialized,encoding='utf-8')
    print('已复核：',dict(Counter(decisions.values())))
    print('保留分组：',report['segment_counts'],'；样本：',report['frame_counts'])
    print('导入结果：',target)


if __name__ == '__main__':
    main()
