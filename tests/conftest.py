from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from filingqa.ingest.chunker import FilingMeta
from filingqa.ingest.html_parser import ParsedFiling, parse_filing

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus"
MINI_GOLDEN = FIXTURES / "mini_golden_SYNTHETIC.jsonl"

DOCS = {
    "acme": ("SYNTHETIC_acme_robotics_10-K_FY2024", "10-K"),
    "bluewater": ("SYNTHETIC_bluewater_foods_10-K_FY2024", "10-K"),
    "cobalt": ("SYNTHETIC_cobalt_energy_10-Q_Q2-2024", "10-Q"),
}


def load_meta(key: str) -> dict:
    stem, _ = DOCS[key]
    return json.loads((CORPUS / f"{stem}.meta.json").read_text())


def parse_doc(key: str) -> ParsedFiling:
    stem, form = DOCS[key]
    return parse_filing((CORPUS / f"{stem}.html").read_bytes(), form)


def filing_meta(key: str) -> FilingMeta:
    m = load_meta(key)
    return FilingMeta(
        accession=m["accession"],
        cik=m["cik"],
        company=m["company"],
        ticker=m["ticker"],
        form_type=m["form_type"],
        filing_date=date.fromisoformat(m["filing_date"]),
        period_of_report=date.fromisoformat(m["period_of_report"]),
        fiscal_year=2024,
        fiscal_period="FY" if m["form_type"] == "10-K" else "Q2",
        source_url=m["source_url"],
        synthetic=True,
    )


@pytest.fixture(scope="session")
def acme() -> ParsedFiling:
    return parse_doc("acme")


@pytest.fixture(scope="session")
def bluewater() -> ParsedFiling:
    return parse_doc("bluewater")


@pytest.fixture(scope="session")
def cobalt() -> ParsedFiling:
    return parse_doc("cobalt")


# --- integration database ---------------------------------------------------------------

TEST_DB_ENV = "FILINGQA_TEST_DATABASE_URL"


def _test_db_url() -> str | None:
    url = os.environ.get(TEST_DB_ENV)
    if not url:
        return None
    # Safety: the fixture wipes the schema, so only ever touch a database named *test*.
    dbname = url.rsplit("/", 1)[-1].split("?")[0]
    if "test" not in dbname:
        raise RuntimeError(f"{TEST_DB_ENV} must point at a database whose name contains 'test'")
    return url


@pytest.fixture(scope="session")
def pg_url() -> str:
    url = _test_db_url()
    if url is None:
        pytest.skip(f"set {TEST_DB_ENV} to run integration tests against Postgres + pgvector")
    import psycopg

    try:
        psycopg.connect(url, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"test database unreachable: {exc}")
    return url


@pytest.fixture(scope="session")
def indexed_db(pg_url: str):
    """Fresh schema + synthetic corpus indexed with offline components (hashing embedder)."""
    from filingqa.config import load_config
    from filingqa.embeddings.hashing import HashingEmbedder
    from filingqa.ingest.pipeline import embed_missing, index_local_dir
    from filingqa.store.db import connect, migrate
    from filingqa.store.repository import ChunkStore

    conn = connect(pg_url)
    conn.execute("DROP SCHEMA IF EXISTS public CASCADE")
    conn.execute("CREATE SCHEMA public")
    applied = migrate(conn)
    assert applied == ["001_init.sql"]
    cfg = load_config("config/offline.yaml", use_env=False)
    cfg.database.url = pg_url
    store = ChunkStore(conn)
    stats = index_local_dir(cfg, store, CORPUS)
    embed = embed_missing(store, HashingEmbedder(dim=512))
    yield {"conn": conn, "store": store, "cfg": cfg, "stats": stats, "embed": embed}
    conn.close()
