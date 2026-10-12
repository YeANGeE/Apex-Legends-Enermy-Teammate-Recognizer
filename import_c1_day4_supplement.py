"""Import prepared DAY4 tasks, without starting any service."""
import import_c1_day3_annotation as shared

if __name__ == '__main__':
    shared.DATA = shared.ROOT/'c1_annotations/day4_supplement_v1'
    shared.TITLE = 'DAY4 E2 E4 E10 E11 E12 补样复核 V1'
    shared.CONFIG_FILE = shared.ROOT/'c1_day4_supplement_label_config.xml'
    shared.DESCRIPTION = '每0.5秒抽帧，DAY4联合n阈值0.10预标注。人工删场景误框、改敌我、补漏检；没有预测也必须查看全图。原五段验证视频不加入此批。'
    shared.MODEL_VERSION = 'day4_supplement_n_conf010_v1'
    shared.main()
