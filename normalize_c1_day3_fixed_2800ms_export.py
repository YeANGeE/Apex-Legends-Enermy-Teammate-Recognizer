"""Validate completed project16 JSON/YOLO ZIP; never train from predictions."""
import json
import normalize_c1_day3_export as shared

if __name__=='__main__':
    shared.EXPORT=shared.ROOT/'c1_yolo_export/DAY3/3.3'
    shared.PREPARED=shared.ROOT/'c1_annotations/day3_fixed_2800ms_v1'
    shared.OUTPUT=shared.ROOT/'c1_yolo_dataset/day3_fixed_2800ms_normalized_v1'
    shared.main()
    path=shared.OUTPUT/'dataset_summary.json'
    summary=json.loads(path.read_text(encoding='utf-8'))
    summary['prelabel_model']='joint_n'
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
