"""End-to-end ingestion: download -> parse -> chunk -> load (+ BM25 stats) -> embed."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from filingqa.config import AppConfig
from filingqa.cost import PriceTable
from filingqa.embeddings.base import Embedder
from filingqa.ingest.chunker import CHUNKER_VERSION, FilingMeta, chunk_filing
from filingqa.ingest.companies import load_tickers, resolve_companies
from filingqa.ingest.edgar import EdgarClient, FilingRef, get_user_agent, load_cached_refs
from filingqa.ingest.fiscal import infer_fiscal_period
from filingqa.ingest.html_parser import PARSER_VERSION, parse_filing
from filingqa.logging_utils import log_event
from filingqa.store.repository import ChunkStore

log = logging.getLogger("filingqa.ingest")


@dataclass
class IndexStats:
    filings_indexed: int = 0
    filings_skipped: int = 0
    filings_failed: int = 0
    chunks: int = 0
    failures: list[str] = field(default_factory=list)


def meta_from_ref(ref: FilingRef) -> FilingMeta:
    """Filing metadata from EDGAR; fiscal labels are filled in after parsing (see index_filing)."""
    return FilingMeta(
        accession=ref.accession,
        cik=ref.cik,
        company=ref.company,
        ticker=ref.ticker,
        form_type=ref.form_type,
        filing_date=ref.filing_date,
        period_of_report=ref.period_of_report,
        fiscal_year=None,
        fiscal_period=None,
        source_url=ref.url,
    )


def index_filing(
    store: ChunkStore,
    raw: bytes,
    meta: FilingMeta,
    cfg: AppConfig,
    fiscal_year_end: str | None = None,
) -> int:
    """Parse, label (fiscal period: inline-XBRL DEI first, date heuristic second), chunk, load."""
    parsed = parse_filing(raw, meta.form_type)
    meta.fiscal_year, meta.fiscal_period = infer_fiscal_period(
        meta.form_type, meta.period_of_report, fiscal_year_end, parsed.dei
    )
    chunks = chunk_filing(parsed, meta, cfg.chunking)
    return store.replace_filing(meta, parsed, chunks, CHUNKER_VERSION)


# --- download ------------------------------------------------------------------------


def _round_robin(refs_by_company: dict[str, list[FilingRef]]) -> list[FilingRef]:
    """Interleave companies so a --max-filings cap still spreads across companies."""
    out: list[FilingRef] = []
    queues = [list(v) for v in refs_by_company.values()]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def download_filings(
    cfg: AppConfig,
    *,
    tickers: list[str] | None = None,
    max_filings: int | None = None,
    client: EdgarClient | None = None,
) -> list[tuple[FilingRef, Path]]:
    ic = cfg.ingest
    own = client is None
    if client is None:
        client = EdgarClient(get_user_agent(), max_requests_per_second=ic.max_requests_per_second)
    try:
        tmap = client.ticker_map()
        companies, unknown = resolve_companies(tickers or load_tickers(ic.companies_file), tmap)
        if unknown:
            log_event(log, "unknown_tickers", logging.WARNING, tickers=unknown)
        by_company: dict[str, list[FilingRef]] = {}
        for co in companies:
            refs = client.list_filings(
                co.cik,
                forms=ic.forms,
                start=ic.start_date,
                end=ic.end_date,
                ticker=co.ticker,
                include_amendments=ic.include_amendments,
            )
            by_company[co.ticker] = refs
            log_event(log, "listed_filings", ticker=co.ticker, cik=co.cik, n=len(refs))
        todo = _round_robin(by_company)
        cap = max_filings or ic.max_filings
        if cap:
            todo = todo[:cap]
        log_event(log, "download_start", filings=len(todo), companies=len(companies))
        done: list[tuple[FilingRef, Path]] = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ic.max_workers) as ex:
            futs = {ex.submit(client.download, r, ic.raw_dir): r for r in todo}
            for i, fut in enumerate(as_completed(futs), start=1):
                ref = futs[fut]
                try:
                    done.append((ref, fut.result()))
                except Exception as exc:  # keep going; report at the end
                    log_event(
                        log,
                        "download_failed",
                        logging.ERROR,
                        accession=ref.accession,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if i % 25 == 0 or i == len(todo):
                    log_event(
                        log,
                        "download_progress",
                        done=i,
                        total=len(todo),
                        elapsed_s=round(time.perf_counter() - t0, 1),
                    )
        return done
    finally:
        if own:
            client.close()


_ARCHIVE_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})/")


def download_for_accessions(
    cfg: AppConfig, accessions: set[str], urls: dict[str, str], client: EdgarClient | None = None
) -> list[tuple[FilingRef, Path]]:
    """Download specific filings (e.g. those referenced by the golden set)."""
    by_cik: dict[int, set[str]] = defaultdict(set)
    for acc in accessions:
        m = _ARCHIVE_RE.search(urls.get(acc, ""))
        if m:
            by_cik[int(m.group(1))].add(acc)
        else:
            log_event(log, "accession_without_url", logging.WARNING, accession=acc)
    own = client is None
    if client is None:
        client = EdgarClient(get_user_agent(), max_requests_per_second=cfg.ingest.max_requests_per_second)
    out: list[tuple[FilingRef, Path]] = []
    try:
        for cik, accs in by_cik.items():
            refs = client.list_filings(
                cik,
                forms=["10-K", "10-Q"],
                start=date(1994, 1, 1),
                end=date(2100, 1, 1),
                include_amendments=True,
            )
            for ref in refs:
                if ref.accession in accs:
                    out.append((ref, client.download(ref, cfg.ingest.raw_dir)))
    finally:
        if own:
            client.close()
    return out


# --- index ---------------------------------------------------------------------------


def index_raw_dir(
    cfg: AppConfig, store: ChunkStore, *, force: bool = False, limit: int | None = None
) -> IndexStats:
    stats = IndexStats()
    cached = load_cached_refs(cfg.ingest.raw_dir)
    if limit:
        cached = cached[:limit]
    t0 = time.perf_counter()
    for i, (ref, path) in enumerate(cached, start=1):
        if not force and store.filing_is_current(ref.accession, PARSER_VERSION, CHUNKER_VERSION):
            stats.filings_skipped += 1
            continue
        try:
            stats.chunks += index_filing(
                store, path.read_bytes(), meta_from_ref(ref), cfg, ref.fiscal_year_end
            )
            stats.filings_indexed += 1
        except Exception as exc:
            stats.filings_failed += 1
            stats.failures.append(f"{ref.accession}: {type(exc).__name__}: {exc}")
            log_event(log, "index_failed", logging.ERROR, accession=ref.accession, error=str(exc))
        if i % 25 == 0 or i == len(cached):
            log_event(
                log,
                "index_progress",
                done=i,
                total=len(cached),
                elapsed_s=round(time.perf_counter() - t0, 1),
            )
    store.refresh_bm25_stats()
    return stats


def index_local_dir(cfg: AppConfig, store: ChunkStore, directory: str | Path) -> IndexStats:
    """Index local HTML files that have a <name>.meta.json sidecar (synthetic fixtures)."""
    stats = IndexStats()
    for html in sorted(Path(directory).glob("*.htm*")):
        sidecar = html.with_name(html.name.rsplit(".", 1)[0] + ".meta.json")
        if not sidecar.is_file():
            continue
        m = json.loads(sidecar.read_text())
        meta = FilingMeta(
            accession=m["accession"],
            cik=int(m["cik"]),
            company=m["company"],
            ticker=m.get("ticker"),
            form_type=m["form_type"],
            filing_date=date.fromisoformat(m["filing_date"]) if m.get("filing_date") else None,
            period_of_report=(
                date.fromisoformat(m["period_of_report"]) if m.get("period_of_report") else None
            ),
            fiscal_year=None,
            fiscal_period=None,
            source_url=m.get("source_url", ""),
            synthetic=bool(m.get("synthetic", False)),
        )
        stats.chunks += index_filing(store, html.read_bytes(), meta, cfg, m.get("fiscal_year_end"))
        stats.filings_indexed += 1
    store.refresh_bm25_stats()
    return stats


# --- embed ---------------------------------------------------------------------------


def embed_missing(
    store: ChunkStore, embedder: Embedder, prices: PriceTable | None = None, page: int = 512
) -> dict:
    prices = prices or PriceTable()
    total = store.count_missing_embeddings(embedder.model)
    done = 0
    tokens = 0
    t0 = time.perf_counter()
    while True:
        rows = store.chunks_missing_embeddings(embedder.model, page)
        if not rows:
            break
        batch = embedder.embed_documents([t for _, t in rows])
        store.insert_embeddings(
            embedder.model,
            embedder.dim,
            [(cid, v) for (cid, _), v in zip(rows, batch.vectors, strict=True)],
        )
        done += len(rows)
        tokens += batch.tokens
        log_event(
            log,
            "embed_progress",
            done=done,
            total=total,
            tokens=tokens,
            elapsed_s=round(time.perf_counter() - t0, 1),
        )
    index = store.ensure_vector_index(embedder.model, embedder.dim)
    cost = prices.cost(getattr(embedder, "model_name", embedder.model), tokens, provider=embedder.provider)
    return {
        "embedded": done,
        "tokens": tokens,
        "cost_usd": round(cost, 4),
        "index": index,
        "model": embedder.model,
        "dim": embedder.dim,
    }
