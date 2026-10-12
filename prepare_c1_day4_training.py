"""Merge project17 corrected error frames with the complete previous train set."""
import prepare_c1_day3_fixed_2800ms_training as shared

if __name__=='__main__':
    shared.BASE=shared.ROOT/'c1_yolo_dataset/day3_joint_fixed_2800ms_v1'
    shared.NEW=shared.ROOT/'c1_yolo_dataset/day4_errors_normalized_v1'
    shared.OUTPUT=shared.ROOT/'c1_yolo_dataset/day4_joint_errors_v1'
    shared.EXPECTED_BASE_COUNTS={'train':1685,'val':62}
    shared.ANNOTATION_SOURCE='day4_targeted_errors_completed_manual_annotation'
    if shared.sha(shared.BASE/'source_manifest.json')!='ef499410b7f86db6a451dca64d159034138b55c382ac73b03741fbce947a451c':
        raise ValueError('Previous dataset manifest changed')
    shared.main()
