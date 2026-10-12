"""Candidate camera-work reduction; preserve detector size and source timestamps."""
import sys
import preview_c1_live_overlay as shared
from benchmark_c1_day4_engine import ENGINE

if __name__=='__main__':
    if any(flag in sys.argv for flag in ['--detector','--tracker','--capture-fps','--ui-poll-ms','--tracking-profile']):
        raise ValueError('V3 fixes the candidate configuration for comparison')
    print('低延迟V3：相对V2仅镜头补偿640/350点→480/200点；YOLO仍1280，检测阈值仍0.10。',flush=True)
    sys.argv=[sys.argv[0],'--detector',str(ENGINE),'--tracker','motion','--capture-fps','120',
              '--ui-poll-ms','2','--latest-capture','--tracking-profile','fast',*sys.argv[1:]]
    shared.main()
