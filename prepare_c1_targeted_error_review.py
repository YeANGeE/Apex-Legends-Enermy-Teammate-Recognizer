"""Extract user-reviewed exact frames and reuse matching inference as drafts."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from PIL import Image
import prepare_c1_day3_recordings as shared

ROOT=Path(__file__).resolve().parent
SOURCE=Path('D:/FPS_AI_Dataset/Apex_YOLO_DAY3')
RUN=ROOT/'runs/c1_tracking/20261010_140327_128754'
OUTPUT=ROOT/'c1_annotations/day3_targeted_errors_v1'
FEEDBACK={
43:{31:'队友0.11正确；右侧队友0.41是背景。',45:'同一中距离人物始终是队友，没有ID时被认成敌人；enemy0.06是背景。',
75:'对照帧：用户指出帧45之后到91之前正常识别队友；同一人物必须标队友。',
91:'同一人物始终是队友；从此帧开始被认成敌人并产生敌我重叠框。',106:'同一队友完全被认成敌人，单一人物多个重叠框；只保留一个队友身体框。'},
47:{10:'enemy是背景。',20:'teammate是枪械配件。',37:'teammate是枪械配件。',44:'teammate是枪械配件。',
105:'同一个枪械配件同时被认成敌人和队友。',109:'enemy0.08是背景。',112:'teammate是枪械配件。',136:'teammate是背景。',241:'teammate0.21是背景；本片另有单一小目标重叠检测。'},
79:{114:'enemy0.05是背景。',137:'检测为队友的真人是敌人；单一人物有重叠框。',148:'enemy0.15和teammate0.22都是背景。',
176:'enemy0.15是倒地队友，应标队友；enemy0.73才是真敌人。目标选择却保持了误识别队友，需独立排查身份和目标保持逻辑。'},
82:{143:'检测到的队友均为敌人；teammate0.06是敌人的可攻击放置技能，按用户明确要求暂标enemy，其他真人各保留一个身体框。',
179:'右侧enemy0.09和teammate0.32都是环境误检。',239:'检测到的enemy均是环境/特效；真实敌人漏检，需补框。',262:'漏掉两个敌人，检测到的teammate是环境。'},
150:{110:'teammate0.21是环境。',122:'teammate是枪械。',128:'检测为teammate的真人是敌人。',140:'enemy是环境，实际敌人漏检。',197:'enemy是环境。'},
178:{74:'teammate0.48实际是敌人。',80:'enemy0.06是环境。',102:'enemy0.13和teammate0.14均是环境。',125:'teammate0.18是环境。',
222:'teammate框住的是游戏字幕。',242:'只有enemy0.06识别主体正确，其他为环境误检。',281:'enemy0.06是环境。',
370:'teammate0.26是环境，与帧281误检的是同一个东西。',392:'enemy0.09是枪械。本片还需检查人物编号跳变。'},
320:{48:'enemy是环境中的树。',52:'teammate是环境。',59:'enemy是同帧48的树。',63:'所有检测均是环境；所选enemy是帧48/59的同一棵树。',
89:'enemy0.13是准心，enemy0.10是环境。',96:'enemy0.26是准心。',103:'enemy0.06是准心，enemy0.27是环境；真实敌人漏检。',
129:'所有teammate是环境；enemy0.11左侧正确、右侧为环境，右侧真实敌人漏检。',258:'enemy和teammate均为环境。',280:'enemy和teammate均为环境。',320:'enemy是环境。'}
}


def main():
    config=json.loads((RUN/'run_config.json').read_text(encoding='utf-8'))
    if config['weights_sha256']!='76cc615ca2c6ced6566fd42cd137ef8744478fb8a9886dd1b9c65459327ace9c':
        raise ValueError('Wrong inference run for user feedback')
    OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'images').mkdir(exist_ok=True);(OUTPUT/'contacts').mkdir(exist_ok=True)
    audits={r['episode']:r for r in json.loads((ROOT/'c1_annotations/day3_v4_prelabel_v1/recording_audit.json').read_text(encoding='utf-8'))}
    records=[];tasks=[];evidence=[]
    for episode,notes in FEEDBACK.items():
        path=SOURCE/f'episode_{episode:06d}';audit=audits[path.name]
        if shared.sha(path/'frames.csv')!=audit['source_frames_sha256'] or shared.sha(path/'metadata.json')!=audit['source_metadata_sha256']:
            raise ValueError('Original source changed')
        with (path/'frames.csv').open(newline='') as handle:frames=list(csv.DictReader(handle))
        logs={r['frame_id']:r for r in (json.loads(s) for s in (RUN/path.name/'tracks.jsonl').read_text(encoding='utf-8').splitlines())}
        indices=sorted(notes)
        targets=[OUTPUT/'images'/f'error_{path.name}_f{i:06d}.png' for i in indices]
        if not all(p.exists() for p in targets):
            if any(p.exists() for p in targets):raise ValueError('Partial batch; inspect before retry')
            temp=OUTPUT/'images'/f'{path.name}_extract_%03d.png'
            selection='+'.join(f'eq(n\\,{i})' for i in indices)
            subprocess.run([shared.resolve_executable('ffmpeg'),'-v','error','-threads','1','-i',str(path/'video.mkv'),
                '-vf','select='+selection,'-fps_mode','vfr','-frames:v',str(len(indices)),'-threads','1','-n',str(temp)],
                capture_output=True,check=True,timeout=180)
            for j,target in enumerate(targets,1):
                source=OUTPUT/'images'/f'{path.name}_extract_{j:03d}.png'
                source.rename(target)
        for index,image in zip(indices,targets,strict=True):
            frame=frames[index];log=logs[index]
            if int(frame['frame_id'])!=index or int(frame['source_present_ns'])!=log['source_present_ns']:raise ValueError('Frame/log timestamp mismatch')
            with Image.open(image) as im:
                im.load();w,h=im.size;pixel=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
                if (w,h)!=(1920,1080):raise ValueError('Wrong extraction dimensions')
            record=dict(episode=path.name,episode_path=str(path),frame_id=index,frame_timestamp_ns=log['source_present_ns'],
                image_path=str(image),image_name=image.name,width=w,height=h,image_sha256=shared.sha(image),pixel_sha256=pixel,
                preview_id=len(records)+1,split='unassigned',match_group='unconfirmed',qpc_session=audit['qpc_session'],
                audit_status=audit['audit_status'],warnings=audit['warnings'],source_metadata_sha256=audit['source_metadata_sha256'],
                source_frames_sha256=audit['source_frames_sha256'],user_feedback=notes[index],
                attackable_skill_exception=episode==82 and index==143)
            regions=[]
            for j,d in enumerate(log['detections']):
                x1,y1,x2,y2=d['box']
                regions.append(dict(id=f'p{record["preview_id"]}_{j}',from_name='label',to_name='image',type='rectanglelabels',
                    original_width=w,original_height=h,image_rotation=0,score=d['score'],value=dict(x=100*x1/w,y=100*y1/h,
                    width=100*(x2-x1)/w,height=100*(y2-y1)/h,rotation=0,rectanglelabels=['enemy' if d['class']==0 else 'teammate'])))
            tasks.append(dict(data=dict(image='/data/local-files/?d='+image.relative_to(ROOT).as_posix(),
                preview_label=f'E{episode:03d} 帧{index}（演示帧号，0起始）｜{notes[index]}'),
                meta={**record,'dataset':OUTPUT.name},predictions=[dict(model_version='fixed2800_reviewed_error_frames',
                    score=max((d['score'] for d in log['detections']),default=0),result=regions)]))
            records.append(record);evidence.append(dict(episode=path.name,frame=index,user_feedback=notes[index],log=log))
        print(f'E{episode:03d}: {len(indices)}张原尺寸帧',flush=True)
    old=json.loads((ROOT/'c1_yolo_dataset/day3_joint_fixed_2800ms_v1/source_manifest.json').read_text(encoding='utf-8'))
    existing={r['pixel_sha256']:r['image_name'] for r in old if r.get('pixel_sha256')}
    duplicates=[dict(image=r['image_name'],existing=existing[r['pixel_sha256']]) for r in records if r['pixel_sha256'] in existing]
    manifest=OUTPUT/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=records:raise ValueError('Existing prepared manifest differs')
    for name,value in [('manifest.json',records),('label_studio_tasks.json',tasks),('review_evidence.json',evidence),('exact_duplicates.json',duplicates),
            ('prelabel_summary.json',dict(images=len(records),predicted_boxes=sum(len(t['predictions'][0]['result']) for t in tasks),
             checkpoint=str(ROOT/'runs/c1_yolo/day3_yolo26n_1280_joint_fixed2800_v1/weights/best.pt'),
             checkpoint_sha256=config['weights_sha256'],
             prediction_source=str(RUN),predictions_are_ground_truth=False,reused_exact_frame_log=True,training_started=False))]:
        shared.save(OUTPUT/name,value)
    shared.OUTPUT=OUTPUT;shared.contacts(records)
    print(f'准备完成：{len(records)}张（46个报告帧+1个正常队友对照），原训练重复{len(duplicates)}张。',flush=True)


if __name__=='__main__':main()
