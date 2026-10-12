"""Normalize new comparison holdout without touching training data."""
import normalize_c1_day2_export as normalize

if __name__ == "__main__":
    normalize.EXPORT=normalize.ROOT/"c1_yolo_export/DAY2/enermy_teammate_val"
    normalize.ROSTER=normalize.ROOT/"c1_annotations/day2_validation_v1/label_studio_tasks.json"
    normalize.OUTPUT=normalize.ROOT/"c1_yolo_dataset/day2_validation_v1"
    normalize.IMAGE_ROOT=normalize.ROOT/"c1_expansion/day2_validation_v1/images"
    normalize.SPLIT="comparison_holdout"
    normalize.main()
