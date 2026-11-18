"""python -m filingqa.store migrate | stats"""

from __future__ import annotations

import argparse
import json
import sys

from filingqa.config import load_config
from filingqa.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filingqa.store")
    ap.add_argument("--config", default=None)
    ap.add_argument("cmd", choices=["migrate", "stats"])
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)

    from filingqa.store.db import connect, migrate
    from filingqa.store.repository import ChunkStore

    conn = connect(cfg.database.url)
    if args.cmd == "migrate":
        applied = migrate(conn)
        print(json.dumps({"applied": applied}))
    else:
        print(json.dumps(ChunkStore(conn).stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
