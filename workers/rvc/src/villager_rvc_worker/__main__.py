"""Process entry point for the persistent RVC worker."""

import sys

from villager_rvc_worker.config import parse_config
from villager_rvc_worker.engine import RvcInferenceEngine
from villager_rvc_worker.protocol import write_error
from villager_rvc_worker.server import WorkerServer


def main() -> None:
    """Build the engine and serve binary protocol frames on standard I/O."""
    try:
        engine = RvcInferenceEngine.load(parse_config())
    except Exception as error:
        write_error(sys.stdout.buffer, error)
        raise
    WorkerServer(engine, sys.stdin.buffer, sys.stdout.buffer, sys.stderr).serve_forever()


if __name__ == "__main__":
    main()
