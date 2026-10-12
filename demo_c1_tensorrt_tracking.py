"""PyCharm entry: candidate tracking regression; does not promote the detector."""
import sys
from pathlib import Path
from demo_c1_video_tracking import main

ENGINE = Path(__file__).resolve().parent / 'runs/c1_tensorrt/20261009_002750_789449/detector.engine'

if __name__ == '__main__':
    main(['--detector', str(ENGINE), *sys.argv[1:]])
