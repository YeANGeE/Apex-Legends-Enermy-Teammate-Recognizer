"""Bounded import of exact user-reported frames; never start Label Studio."""
import import_c1_day3_annotation as shared

if __name__=='__main__':
    shared.DATA=shared.ROOT/'c1_annotations/day3_targeted_errors_v1'
    shared.TITLE='DAY3 逐帧错误补标 V1'
    shared.CONFIG_FILE=shared.ROOT/'c1_targeted_error_label_config.xml'
    shared.DESCRIPTION='46个用户报告帧+1个队友正常对照。复用同帧预测，人工删除误检、补漏、改敌我及重叠框。E82帧143可攻击放置技能按用户指示暂标enemy，metadata记录例外。'
    shared.MODEL_VERSION='fixed2800_reviewed_error_frames'
    shared.main()
