"""V3 plus scoped 1ms Windows timer request; no model/tracker changes."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile']):
        raise ValueError('V4 fixes candidate settings')
    print('低延迟V4：相对V3只请求1ms等待定时精度；退出恢复，并记录真实显示回调间隔。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120','--ui-poll-ms','2',
              '--latest-capture','--tracking-profile','fast','--high-resolution-timer',*sys.argv[1:]]
    shared.main()
