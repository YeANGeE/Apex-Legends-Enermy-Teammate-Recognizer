"""Validate completed project17 exact-error-frame annotation exports."""
import json
import normalize_c1_day3_export as shared

if __name__=='__main__':
    shared.EXPORT=shared.ROOT/'c1_yolo_export/DAY4'
    shared.PREPARED=shared.ROOT/'c1_annotations/day3_targeted_errors_v1'
    shared.OUTPUT=shared.ROOT/'c1_yolo_dataset/day4_errors_normalized_v1'
    shared.main()
    path=shared.OUTPUT/'dataset_summary.json'
    summary=json.loads(path.read_text(encoding='utf-8'))
    summary['prelabel_model']='joint_fixed2800'
    summary['attackable_skill_policy']='User explicitly treats E82 frame143 enemy deployable as enemy; source metadata records exception'
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
