"""User-started offline-range shadow preview using the accepted C1 V6 capture path."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE
from c2_shadow import WEIGHTS

if __name__ == '__main__':
    fixed = ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile',
             '--event-wakeup','--c2-weights','--capture-color']
    if any(arg.split('=')[0] in fixed for arg in sys.argv[1:]):
        raise ValueError('Shadow preview fixes C1 V6 and the accepted C2 baseline')
    print('C2只读影子：DAY4检测＋V6采集＋联合C2；仅显示RX/RY/RB开火/LB瞄准建议，不发送手柄。', flush=True)
    sys.argv = [sys.argv[0], '--detector', str(ENGINE), '--tracker', 'motion',
                '--capture-fps', '120', '--ui-poll-ms', '2', '--latest-capture',
                '--tracking-profile', 'fast', '--high-resolution-timer', '--capture-timing',
                '--capture-downscale', '--c2-weights', str(WEIGHTS), *sys.argv[1:]]
    shared.main()
