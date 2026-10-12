"""V4 plus capture-event inference wakeup; same detector/tracker/timing."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile']):
        raise ValueError('V5 fixes candidate settings')
    print('低延迟V5：相对V4只改为采集事件唤醒推理，不再轮询等待；其他参数保持。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120','--ui-poll-ms','2',
              '--latest-capture','--tracking-profile','fast','--high-resolution-timer','--event-wakeup',*sys.argv[1:]]
    shared.main()
