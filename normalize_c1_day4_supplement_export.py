"""Validate project18 manual JSON/YOLO ZIP against the prepared source roster."""
import json
import normalize_c1_day3_export as shared

if __name__ == '__main__':
    shared.EXPORT = shared.ROOT/'c1_yolo_export/DAY4/4.1'
    shared.PREPARED = shared.ROOT/'c1_annotations/day4_supplement_v1'
    shared.OUTPUT = shared.ROOT/'c1_yolo_dataset/day4_supplement_normalized_v1'
    shared.main()
    path = shared.OUTPUT/'dataset_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['prelabel_model'] = 'day4_joint_errors_n_conf010'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
