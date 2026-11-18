"""Ingestion CLI.

python -m filingqa.ingest download [--tickers AAPL,MSFT] [--max-filings N]
python -m filingqa.ingest index [--force] [--limit N]
python -m filingqa.ingest embed
python -m filingqa.ingest all                 # download + index + embed
python -m filingqa.ingest local DIR           # index local fixtures (with .meta.json sidecars)
python -m filingqa.ingest from-golden         # download only filings the golden set cites
python -m filingqa.ingest stats
"""

from __future__ import annotations

import argparse
import json
import sys

from filingqa.config import load_config
from filingqa.logging_utils import setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filingqa.ingest")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--tickers", default=None, help="comma-separated; default: companies file")
    d.add_argument("--max-filings", type=int, default=None)
    i = sub.add_parser("index")
    i.add_argument("--force", action="store_true", help="re-index filings already loaded")
    i.add_argument("--limit", type=int, default=None)
    sub.add_parser("embed")
    a = sub.add_parser("all")
    a.add_argument("--tickers", default=None)
    a.add_argument("--max-filings", type=int, default=None)
    loc = sub.add_parser("local")
    loc.add_argument("directory")
    g = sub.add_parser("from-golden")
    g.add_argument("--golden", default=None)
    sub.add_parser("stats")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)

    from filingqa.components import load_prices
    from filingqa.ingest import pipeline
    from filingqa.store.db import connect
    from filingqa.store.repository import ChunkStore

    def store() -> ChunkStore:
        return ChunkStore(connect(cfg.database.url))

    if args.cmd in ("download", "all"):
        tickers = args.tickers.split(",") if args.tickers else None
        done = pipeline.download_filings(cfg, tickers=tickers, max_filings=args.max_filings)
        print(json.dumps({"downloaded_or_cached": len(done)}))
    if args.cmd in ("index", "all"):
        st = pipeline.index_raw_dir(
            cfg, store(), force=getattr(args, "force", False), limit=getattr(args, "limit", None)
        )
        print(json.dumps(st.__dict__, default=str))
    if args.cmd in ("embed", "all"):
        from filingqa.embeddings import make_embedder

        print(json.dumps(pipeline.embed_missing(store(), make_embedder(cfg.embedding), load_prices(cfg))))
    if args.cmd == "local":
        st = pipeline.index_local_dir(cfg, store(), args.directory)
        print(json.dumps(st.__dict__, default=str))
    if args.cmd == "from-golden":
        from filingqa.eval.golden import load_golden

        items = load_golden(args.golden or cfg.eval.golden_path)
        accs: set[str] = set()
        urls: dict[str, str] = {}
        for it in items:
            if it.counts_as_verified():
                for gp in it.gold:
                    accs.add(gp.accession)
                    if gp.source_url:
                        urls[gp.accession] = gp.source_url
        done = pipeline.download_for_accessions(cfg, accs, urls)
        print(json.dumps({"referenced": len(accs), "downloaded_or_cached": len(done)}))
    if args.cmd == "stats":
        print(json.dumps(store().stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
