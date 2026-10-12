"""User-started physical-controller calibration capture, no virtual input."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE
from c2_shadow import WEIGHTS

if __name__ == '__main__':
    fixed = ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile',
             '--event-wakeup','--c2-weights','--capture-color','--c2-geometry-bias','--record-xinput']
    if any(arg.split('=')[0] in fixed for arg in sys.argv[1:]):
        raise ValueError('Calibration fixes accepted capture and read-only physical input recording')
    print('离线校准：请站定，选静止假人，暂不开火。你操作真实手柄；记录约4000Hz读取和目标位置，不输出控制。',flush=True)
    print('小幅向左/右、上/下推动，每次短推后回中；让假人始终留在画面内。hip不瞄准，ads按住LB。',flush=True)
    sys.argv = [sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120',
                '--ui-poll-ms','2','--latest-capture','--tracking-profile','fast',
                '--high-resolution-timer','--capture-timing','--capture-downscale',
                '--c2-weights',str(WEIGHTS),'--c2-geometry-bias','--record-xinput',*sys.argv[1:]]
    shared.main()
