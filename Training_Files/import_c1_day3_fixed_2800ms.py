"""Import a separate annotation project; never start a server."""
import import_c1_day3_annotation as shared

if __name__=='__main__':
    shared.DATA=shared.ROOT/'c1_annotations/day3_fixed_2800ms_v1'
    shared.TITLE='DAY3 固定2.8秒 联合n预标注复核 V1'
    shared.CONFIG_FILE=shared.ROOT/'c1_day3_fixed_2800ms_label_config.xml'
    shared.DESCRIPTION='每段DAY3原生时间2.8秒附近一帧；联合n预标，人工完整复核。用于补充训练，原验证保持。准心本身不标，覆盖真人时仍标真人。'
    shared.MODEL_VERSION='joint_n_fixed2800_conf005'
    shared.main()
