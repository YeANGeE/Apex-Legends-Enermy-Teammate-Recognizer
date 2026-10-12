"""User-started V4 with causal capture-stage diagnostics; not a new speed claim."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile','--event-wakeup']):
        raise ValueError('Diagnosis uses the accepted V4 baseline')
    print('采集延迟诊断：保持V4；新增源时间→拷贝→交付→推理开始计时，不改变录制对齐时间。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120','--ui-poll-ms','2',
              '--latest-capture','--tracking-profile','fast','--high-resolution-timer','--capture-timing',*sys.argv[1:]]
    shared.main()
