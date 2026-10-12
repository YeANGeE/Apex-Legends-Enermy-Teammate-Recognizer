"""User-started opt-in TRT/motion/120Hz read-only overlay candidate."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms']):
        raise ValueError('This comparison entry pins TRT, motion, 120Hz capture and 2ms UI polling')
    print('低延迟候选：当前模型TensorRT FP16，motion跟踪，120Hz采集，2ms显示轮询；检测参数保持。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120','--ui-poll-ms','2',*sys.argv[1:]]
    shared.main()
