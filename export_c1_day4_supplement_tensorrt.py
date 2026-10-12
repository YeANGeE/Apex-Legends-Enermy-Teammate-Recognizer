"""Mechanical export of current candidate, preserving PT and NMS branch."""
import export_c1_tensorrt as shared
from demo_c1_day4_supplement_tracking import WEIGHTS, EXPECTED_SHA

if __name__ == '__main__':
    if shared.sha256(WEIGHTS) != EXPECTED_SHA:
        raise ValueError('Current candidate weight changed')
    shared.WEIGHTS = WEIGHTS
    shared.main()
