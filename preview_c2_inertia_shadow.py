"""Offline-range read-only BC + bias + 80ms-response trajectory preview."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE
from c2_shadow import WEIGHTS

if __name__ == '__main__':
    fixed = ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile',
             '--event-wakeup','--c2-weights','--capture-color','--c2-geometry-bias']
    if any(arg.split('=')[0] in fixed for arg in sys.argv[1:]):
        raise ValueError('Inertia preview fixes accepted perception and base policy')
    response = [] if any(arg.split('=')[0]=='--c2-inertia-ms' for arg in sys.argv[1:]) else ['--c2-inertia-ms','80']
    print('惯性只读预览：C2＋bias＋80ms候选响应；请求1ms轨迹更新，记录真实周期，不发送手柄。',flush=True)
    sys.argv = [sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120',
                '--ui-poll-ms','2','--latest-capture','--tracking-profile','fast',
                '--high-resolution-timer','--capture-timing','--capture-downscale',
                '--c2-weights',str(WEIGHTS),'--c2-geometry-bias',*response,*sys.argv[1:]]
    shared.main()
