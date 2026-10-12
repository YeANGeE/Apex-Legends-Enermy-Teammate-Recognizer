"""Read-only offline-range base BC + additive geometry correction."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE
from c2_shadow import WEIGHTS

if __name__ == '__main__':
    fixed = ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile',
             '--event-wakeup','--c2-weights','--capture-color','--c2-geometry-bias']
    if any(arg.split('=')[0] in fixed for arg in sys.argv[1:]):
        raise ValueError('Candidate fixes C1 V6 and the current joint BC baseline')
    print('几何bias只读候选：保留C2原始预测，追加目标中心差值修正；增益尚未强化学习，不发送手柄。',flush=True)
    sys.argv = [sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120',
                '--ui-poll-ms','2','--latest-capture','--tracking-profile','fast',
                '--high-resolution-timer','--capture-timing','--capture-downscale',
                '--c2-weights',str(WEIGHTS),'--c2-geometry-bias',*sys.argv[1:]]
    shared.main()
