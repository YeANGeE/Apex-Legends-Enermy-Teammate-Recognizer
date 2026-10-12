"""Audited mechanical merge, preserving old dataset and validation files."""
import hashlib
import json
import shutil
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from normalize_c1_day3_export import sha

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'c1_yolo_dataset/day3_v4_targeted_v1'
NEW=ROOT/'c1_yolo_dataset/day3_fixed_2800ms_normalized_v1'
OUTPUT=ROOT/'c1_yolo_dataset/day3_joint_fixed_2800ms_v1'
EXPECTED_BASE_COUNTS={'train':1384,'val':62}
ANNOTATION_SOURCE='day3_fixed2800_completed_manual_annotation'
NEW_ENCOUNTER_ID='day3_match_boundaries_unconfirmed_all_train'
NEW_SOURCE_DOMAIN='day3_real_match'
NEW_IMAGE_PREFIX=''


def pixels(path):
    with Image.open(path) as im:return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve prepared version')
    old=json.loads((BASE/'source_manifest.json').read_text(encoding='utf-8'))
    fresh=json.loads((NEW/'source_manifest.json').read_text(encoding='utf-8'))
    if Counter(r['split'] for r in old)!=EXPECTED_BASE_COUNTS:raise ValueError('Old roster differs')
    rows=[];train_pixels={};val_pixels=set();duplicates=[]
    for r in old:
        image=BASE/'images'/r['split']/r['image_name']
        label=BASE/'labels'/r['split']/(image.stem+'.txt')
        if sha(image)!=r['image_sha256'] or sha(label)!=r['label_sha256']:raise ValueError('Old data changed')
        pixel=r.get('pixel_sha256') or pixels(image)
        row={**r,'pixel_sha256':pixel,'source_image':str(image),'source_label':str(label)}
        if r['split']=='val':val_pixels.add(pixel)
        else:train_pixels.setdefault(pixel,[]).append(row)
        rows.append(row)
    added=0;replaced=0
    for r in fresh:
        if not r['eligible']:continue
        image,label=Path(r['normalized_image']),Path(r['normalized_label'])
        if sha(image)!=r['image_sha256'] or sha(label)!=r['label_sha256']:raise ValueError('New data changed')
        pixel=r['pixel_sha256']
        if pixel in val_pixels:raise ValueError('New image duplicates validation; do not train')
        prior=train_pixels.get(pixel,[])
        row={**r,'split':'train','encounter_id':NEW_ENCOUNTER_ID,
             'source_domain':NEW_SOURCE_DOMAIN,'source_image':str(image),'source_label':str(label),
             'scene_type':'real_match','annotation_source':ANNOTATION_SOURCE}
        if NEW_IMAGE_PREFIX:
            row.update(original_image_name=r['image_name'], image_name=NEW_IMAGE_PREFIX+r['image_name'])
        if prior:
            if len(prior)!=1:raise ValueError('Ambiguous duplicate annotations')
            previous=prior[0]
            # A later review of the identical source frame supersedes the old label
            # only in this new version. Preserve the old version and audit both labels.
            if row['annotation_updated_at']<=previous.get('annotation_updated_at',''):
                raise ValueError('Duplicate review is not newer')
            if Counter(b['class_id'] for b in row['boxes'])!=Counter(b['class_id'] for b in previous.get('boxes',[])):
                raise ValueError('Duplicate identity/count conflict requires manual review')
            duplicates.append(dict(old_image=previous['image_name'],new_image=row['image_name'],
                old_boxes=previous.get('boxes'),new_boxes=row['boxes'],policy='newer human box geometry in new version only'))
            rows.remove(previous);replaced+=1
        else:added+=1
        rows.append(row);train_pixels[pixel]=[row]
    if len({r['image_name'] for r in rows})!=len(rows):raise ValueError('Filename collision')
    for field in ('image_sha256','pixel_sha256','encounter_id'):
        train={r[field] for r in rows if r['split']=='train'}
        val={r[field] for r in rows if r['split']=='val'}
        if train&val:raise ValueError(f'Train/val overlap: {field}')
    OUTPUT.mkdir(parents=True);stats=defaultdict(Counter)
    for i,r in enumerate(rows,1):
        image=OUTPUT/'images'/r['split']/r['image_name'];label=OUTPUT/'labels'/r['split']/(image.stem+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['source_image'],image);shutil.copy2(r['source_label'],label)
        if sha(image)!=r['image_sha256'] or sha(label)!=r['label_sha256']:raise ValueError('Copy verification failed')
        lines=label.read_text(encoding='utf-8').splitlines()
        stats[r['split']].update(images=1,negative_images=int(not lines),enemy_boxes=sum(s.startswith('0 ') for s in lines),
                                teammate_boxes=sum(s.startswith('1 ') for s in lines))
        r.update(normalized_image=str(image),normalized_label=str(label))
        if i%200==0:print(f'合并并核对 {i}/{len(rows)}',flush=True)
    summary=dict(stage='Stage 1 — Local Combat Policy',images=dict(Counter(r['split'] for r in rows)),
        split_statistics={k:dict(v) for k,v in stats.items()},new_usable_images=sum(r['eligible'] for r in fresh),
        new_unique_images_added=added,old_duplicate_images_replaced=replaced,duplicate_reviews=duplicates,
        baseline_manifest_sha256=sha(BASE/'source_manifest.json'),new_manifest_sha256=sha(NEW/'source_manifest.json'),
        all_new_usable_images_train=True,original_validation_preserved=True,test_images_used=0,training_started=False,
        warning='Old validation monitors fitting; seven DAY3 diagnostic clips are training exposed; fresh unseen videos needed for generalization')
    (OUTPUT/'data.yaml').write_text(f'path: {OUTPUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: enemy\n  1: teammate\n',encoding='utf-8')
    for name,value in [('source_manifest.json',rows),('dataset_summary.json',summary)]:
        (OUTPUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
