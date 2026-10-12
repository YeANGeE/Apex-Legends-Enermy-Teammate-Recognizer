"""V6 plus BGRA capture then alpha drop at 1080p, not 4K BGR conversion."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile','--event-wakeup','--capture-color']):
        raise ValueError('V7 fixes candidate comparison settings')
    print('低延迟V7：相对V6只改BGRA采集，缩为1080p后转BGR；模型与跟踪参数保持。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120','--ui-poll-ms','2',
              '--latest-capture','--tracking-profile','fast','--high-resolution-timer','--capture-timing',
              '--capture-downscale','--capture-color','BGRA',*sys.argv[1:]]
    shared.main()
