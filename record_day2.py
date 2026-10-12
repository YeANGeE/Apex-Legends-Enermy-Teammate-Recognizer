"""PyCharm entry point; unlimited rolling capture until explicit Ctrl+C."""
from pathlib import Path
import sys

from fps_recorder.main import main


if __name__ == "__main__":
    # Multiprocessing children import this file without executing the recorder.
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv):
        sys.argv.extend(["--config", str(Path(__file__).with_name("config.yaml"))])
    raise SystemExit(main())
