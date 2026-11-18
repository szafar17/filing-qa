"""Integration tests against a real Postgres + pgvector database.

Run with e.g.  FILINGQA_TEST_DATABASE_URL=postgresql://filingqa:filingqa@localhost:5433/filingqa_test
(`docker compose up -d` then `createdb`), or let CI provide the pgvector service container.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import pytest
from fastapi.testclient import TestClient

from filingqa.answer.service import AnswerService
from filingqa.embeddings.hashing import HashingEmbedder
from filingqa.eval.golden import load_golden
from filingqa.eval.runner import EvalOptions, resolve_gold, run_eval
from filingqa.ingest.pipeline import embed_missing, index_local_dir
from filingqa.llm.fake import FakeLLM
from filingqa.retrieval.rerank import LexicalReranker
from filingqa.retrieval.retriever import Retriever
from filingqa.store.db import migrate
from filingqa.store.repository import SearchFilters, index_name_for

from .conftest import CORPUS, MINI_GOLDEN, load_meta

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def comps(indexed_db):
    from filingqa.components import Components
    from filingqa.cost import PriceTable

    cfg = indexed_db["cfg"]
    store = indexed_db["store"]
    embedder = HashingEmbedder(dim=512)
    retriever = Retriever(store, embedder, LexicalReranker(), cfg.retrieval, PriceTable())
    llm = FakeLLM()
    service = AnswerService(retriever, llm, cfg.refusal, PriceTable())
    return Components(cfg, store, embedder, retriever.reranker, retriever, llm, PriceTable(), service)


def test_schema_and_index_load(indexed_db):
    conn, store = indexed_db["conn"], indexed_db["store"]
    assert migrate(conn) == []  # idempotent
    st = store.stats()
    assert st["synthetic_filings"] == 3 and st["filings"] == 0  # synthetic never counts as real
    assert st["chunks"] == indexed_db["stats"].chunks > 30
    assert st["table_chunks"] >= 5
    assert st["embeddings_by_model"] == {"hashing-512": st["chunks"]}
    assert not store.bm25_stats_stale()
    n_null = conn.execute("SELECT count(*) FROM chunks WHERE doc_len IS NULL OR doc_len = 0").fetchone()[0]
    assert n_null == 0


def test_chunk_text_matches_stored_filing_text(indexed_db):
    conn = indexed_db["conn"]
    bad = conn.execute(
        """SELECT count(*) FROM chunks c JOIN filing_texts t USING (accession)
           WHERE substr(t.text, c.char_start + 1, c.char_end - c.char_start) <> c.text"""
    ).fetchone()[0]
    assert bad == 0


@pytest.mark.parametrize(
    "query",
    [
        "What was Acme Robotics total revenue in fiscal 2024?",
        "inventories June 30 2024 Bluewater",
        "interest rate swaps hedge variable-rate project debt",
        "ransomware attack logistics provider",
    ],
)
def test_sql_bm25_matches_python_reference(indexed_db, query):
    conn, store = indexed_db["conn"], indexed_db["store"]
    k1, b = 1.2, 0.75
    docs: dict[int, dict[str, int]] = defaultdict(dict)
    for cid, lex, tf in conn.execute(
        "SELECT c.id, u.lexeme, coalesce(array_length(u.positions, 1), 1) "
        "FROM chunks c CROSS JOIN LATERAL unnest(c.tsv) AS u"
    ).fetchall():
        docs[cid][lex] = tf
    dl = {cid: sum(t.values()) for cid, t in docs.items()}
    n, avgdl = len(docs), sum(dl.values()) / len(dl)
    df = Counter(lex for t in docs.values() for lex in t)
    q = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english', %s)))", (query,)
        ).fetchall()
    ]
    ref = {}
    for cid, t in docs.items():
        s = sum(
            math.log(1 + (n - df[x] + 0.5) / (df[x] + 0.5))
            * t[x]
            * (k1 + 1)
            / (t[x] + k1 * (1 - b + b * dl[cid] / avgdl))
            for x in q
            if x in t
        )
        if s > 0:
            ref[cid] = s
    expected = sorted(ref.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    got = store.bm25_search(query, 10, k1=k1, b=b)
    assert [c for c, _ in got] == [c for c, _ in expected]
    assert [s for _, s in got] == pytest.approx([s for _, s in expected], rel=1e-9)


def test_vector_search_self_retrieval_and_hnsw_index(indexed_db):
    conn, store = indexed_db["conn"], indexed_db["store"]
    emb = HashingEmbedder(dim=512)
    cid, text = conn.execute(
        "SELECT id, context_header || E'\\n' || text FROM chunks ORDER BY id LIMIT 1 OFFSET 7"
    ).fetchone()
    hits = store.vector_search(emb.model, emb.dim, emb.embed_query(text).vectors[0], 5)
    assert hits[0][0] == cid and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1))
    name = index_name_for(emb.model, emb.dim)
    assert conn.execute("SELECT count(*) FROM pg_indexes WHERE indexname = %s", (name,)).fetchone()[0] == 1
    with conn.transaction():
        # 60 rows: the planner rightly prefers a sort; forbid it to prove the index applies
        conn.execute("SET LOCAL enable_seqscan = off")
        conn.execute("SET LOCAL enable_sort = off")
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT chunk_id FROM chunk_embeddings WHERE model = 'hashing-512' "
                "ORDER BY embedding::vector(512) <=> %s::vector(512) LIMIT 5",
                ("[" + ",".join(["0.01"] * 512) + "]",),
            ).fetchall()
        )
    assert name in plan  # the partial HNSW expression index is usable by the planner


def test_filters_restrict_results(indexed_db):
    store = indexed_db["store"]
    emb = HashingEmbedder(dim=512)
    acme = load_meta("acme")
    f = SearchFilters(tickers=["zzacm"])
    ids = [c for c, _ in store.bm25_search("revenue interest rates", 20, f)]
    ids += [
        c for c, _ in store.vector_search(emb.model, emb.dim, emb.embed_query("revenue").vectors[0], 20, f)
    ]
    recs = store.get_chunks(ids)
    assert recs and {r.accession for r in recs.values()} == {acme["accession"]}
    f2 = SearchFilters(form_types=["10-Q"], fiscal_years=[2024])
    recs2 = store.get_chunks([c for c, _ in store.bm25_search("revenue", 20, f2)])
    assert {r.form_type for r in recs2.values()} == {"10-Q"}


def test_all_retrieval_modes(comps):
    q = "What was Bluewater Foods' inventory balance at June 30, 2024?"
    for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        rr = comps.retriever.retrieve(q, mode=mode, k=5)
        assert 0 < len(rr.hits) <= 5
        assert [h.rank for h in rr.hits] == list(range(1, len(rr.hits) + 1))
        assert rr.total_ms > 0
        if mode == "hybrid_rerank":
            assert rr.evidence_kind == "rerank" and all("rerank" in h.scores for h in rr.hits)
            assert [h.score for h in rr.hits] == sorted([h.score for h in rr.hits], reverse=True)
        if mode == "hybrid":
            assert all("rrf" in h.scores for h in rr.hits) and rr.evidence_kind == "vector"
    top = comps.retriever.retrieve(q, mode="hybrid_rerank", k=3).hits
    assert any("Inventories | June 30, 2024: 412.8" in h.chunk.text for h in top)


def test_reindex_is_idempotent_and_cascades(indexed_db):
    conn, store, cfg = indexed_db["conn"], indexed_db["store"], indexed_db["cfg"]
    before = store.stats()
    index_local_dir(cfg, store, CORPUS)
    after = store.stats()
    assert after["chunks"] == before["chunks"]
    # re-indexing replaced the chunks, so their embeddings were cascaded away; re-embed
    assert store.count_missing_embeddings("hashing-512") == after["chunks"]
    res = embed_missing(store, HashingEmbedder(dim=512))
    assert res["embedded"] == after["chunks"] and store.count_missing_embeddings("hashing-512") == 0
    dupes = conn.execute("SELECT count(*) - count(DISTINCT (lexeme, chunk_id)) FROM chunk_terms").fetchone()[
        0
    ]
    assert dupes == 0


def test_answer_service_end_to_end(comps, indexed_db):
    conn = indexed_db["conn"]
    out = comps.service.ask(
        "How many megawatts of solar generation did Cobalt Energy have in "
        "commercial operation as of June 30, 2024?"
    )
    r = out.response
    assert not r.refused and "1,240 megawatts" in r.answer
    c = r.citations[0]
    stored = conn.execute(
        "SELECT substr(text, %s + 1, %s) FROM filing_texts WHERE accession = %s",
        (c.char_start, c.char_end - c.char_start, c.chunk_id.split(":")[0]),
    ).fetchone()[0]
    assert stored == c.quote  # the citation points at the exact span of the filing text
    refused = comps.service.ask("What is the airspeed velocity of an unladen swallow?").response
    assert refused.refused and refused.refusal_reason


def test_eval_runner_smoke(comps):
    items = load_golden(MINI_GOLDEN)
    res = run_eval(
        comps,
        items,
        EvalOptions(
            modes=["vector", "bm25", "hybrid", "hybrid_rerank"],
            answer_modes=["hybrid_rerank"],
            ks=[1, 5, 10],
            official=False,
            progress_every=1000,
        ),
        FakeLLM(model="fake-judge"),
        {"path": str(MINI_GOLDEN), "n_human_verified": 0},
    )
    assert res["official"] is False and res["golden"]["n_skipped_unresolved_gold"] == 0
    assert res["golden"]["n_answerable"] == 13 and res["golden"]["n_unanswerable"] == 3
    for mode, m in res["retrieval"].items():
        assert m["n"] == 13
        assert 0 <= m["recall@1"] <= m["recall@5"] <= m["recall@10"] <= 1, mode
        assert 0 <= m["mrr"] <= 1 and m["latency_ms_p95"] >= m["latency_ms_p50"] > 0
    a = res["answer"]["hybrid_rerank"]
    assert a["n"] == 16 and 0 <= a["refusal_accuracy"] <= 1 and 0 <= a["false_refusal_rate"] <= 1
    assert a["faithfulness"] is None or 0 <= a["faithfulness"] <= 1
    assert len(res["per_item"]) == 16
    assert res["threshold_sweep"]["hybrid_rerank"]["evidence_kind"] == "rerank"


def test_gold_resolution_relocates_stale_offsets(indexed_db):
    store = indexed_db["store"]
    item = load_golden(MINI_GOLDEN)[0]
    good = resolve_gold(item, store.filing_text)
    shifted = item.model_copy(deep=True)
    shifted.gold[0].char_start += 7
    shifted.gold[0].char_end += 7
    assert resolve_gold(shifted, store.filing_text) == good  # relocated by quote
    missing = item.model_copy(deep=True)
    missing.gold[0].quote = "this sentence is not in the filing"
    missing.gold[0].char_start = None
    assert resolve_gold(missing, store.filing_text) is None


def test_draft_from_real_chunks(indexed_db):
    from filingqa.eval.draft import draft_from_chunk, draft_unanswerable

    store = indexed_db["store"]
    chunks = store.sample_chunks(6, include_synthetic=True, min_chars=200)
    assert chunks and not store.sample_chunks(6, include_synthetic=False)  # synthetic excluded by default
    made = 0
    for ch in chunks:
        d, status = draft_from_chunk(FakeLLM(), ch)
        if d is None:
            continue
        made += 1
        assert d.labeled_by == "llm-draft" and d.verified is False and d.status == "pending"
        g = d.gold[0]
        text = store.filing_text(g.accession)
        assert text[g.char_start : g.char_end] == g.quote
    assert made > 0
    un = draft_unanswerable(FakeLLM(), chunks[0])
    assert un is not None and not un.answerable and un.gold == []


def test_eval_cli_smoke_and_gate(indexed_db, pg_url, tmp_path, monkeypatch, capsys):
    from filingqa.eval.__main__ import main

    monkeypatch.setenv("FILINGQA_DATABASE_URL", pg_url)
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"metrics": {"retrieval.bm25.recall@10": {"min": 0.5}}}))
    code = main(
        [
            "--config",
            "config/offline.yaml",
            "--smoke",
            "--results-dir",
            str(tmp_path),
            "--thresholds",
            str(th),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "gate" in out and "PASS" in out
    saved = json.loads((tmp_path / "smoke_latest.json").read_text())
    assert saved["official"] is False and (tmp_path / "smoke_latest.md").is_file()
    th.write_text(json.dumps({"metrics": {"retrieval.bm25.recall@10": {"min": 1.01}}}))
    assert (
        main(
            [
                "--config",
                "config/offline.yaml",
                "--smoke",
                "--results-dir",
                str(tmp_path),
                "--thresholds",
                str(th),
            ]
        )
        == 1
    )


def test_api_with_real_store(comps):
    from filingqa.api.app import create_app

    with TestClient(create_app(service_factory=lambda: (comps.service, None))) as client:
        body = client.post(
            "/ask",
            json={
                "question": "What share of Bluewater Foods' net sales came from its largest customer?",
                "filters": {"tickers": ["ZZBWF"]},
            },
        ).json()
    assert body["refused"] is False and body["citations"][0]["ticker"] == "ZZBWF"


def test_official_eval_refuses_index_with_synthetic_filings(
    indexed_db, pg_url, tmp_path, monkeypatch, capsys
):
    from datetime import UTC, datetime

    from filingqa.eval.__main__ import main
    from filingqa.eval.golden import GoldenItem, GoldPassage, write_jsonl

    monkeypatch.setenv("FILINGQA_DATABASE_URL", pg_url)
    g = tmp_path / "golden.jsonl"
    write_jsonl(
        g,
        [
            GoldenItem(
                id="q1",
                question="What was revenue?",
                labeled_by="human",
                verified=True,
                verified_at=datetime(2026, 1, 1, tzinfo=UTC),
                gold=[GoldPassage(accession="0009900001-25-000004", quote="Total revenue")],
            )
        ],
    )
    # real (non-fake) providers, but retrieval-only and no reranker: nothing calls an API
    code = main(
        [
            "--config",
            "config/default.yaml",
            "--golden",
            str(g),
            "--min-verified",
            "1",
            "--retrieval-only",
            "--modes",
            "bm25",
        ]
    )
    assert code == 2
    assert "synthetic test filing" in capsys.readouterr().out


def test_download_then_index_pipeline_with_mocked_edgar(indexed_db, tmp_path):
    """ticker -> CIK -> submissions -> download -> parallel parse/chunk -> Postgres."""
    import httpx

    from filingqa.ingest.edgar import EdgarClient, RateLimiter
    from filingqa.ingest.pipeline import download_filings, index_raw_dir

    store, conn, cfg = indexed_db["store"], indexed_db["conn"], indexed_db["cfg"]
    acc = "0009900001-25-999999"
    html = (CORPUS / "SYNTHETIC_acme_robotics_10-K_FY2024.html").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(
                200, json={"0": {"cik_str": 9900001, "ticker": "ZZACM", "title": "ACME ROBOTICS CORP"}}
            )
        if url.endswith("CIK0009900001.json"):
            return httpx.Response(
                200,
                json={
                    "name": "ACME ROBOTICS CORP",
                    "fiscalYearEnd": "1231",
                    "filings": {
                        "recent": {
                            "accessionNumber": [acc, "0009900001-25-000001"],
                            "form": ["10-K", "8-K"],
                            "filingDate": ["2025-02-20", "2025-01-05"],
                            "reportDate": ["2024-12-31", ""],
                            "primaryDocument": ["acme-20241231.htm", "8k.htm"],
                        },
                        "files": [],
                    },
                },
            )
        if url.endswith("/acme-20241231.htm"):
            return httpx.Response(200, content=html)
        return httpx.Response(404)

    cfg2 = cfg.model_copy(deep=True)
    cfg2.ingest.raw_dir = str(tmp_path / "raw")
    from datetime import date

    cfg2.ingest.start_date, cfg2.ingest.end_date = date(2024, 7, 1), date(2025, 6, 30)
    client = EdgarClient(
        "Test Runner tests@filingqa.invalid",
        transport=httpx.MockTransport(handler),
        limiter=RateLimiter(10, sleep=lambda s: None),
    )
    try:
        done = download_filings(cfg2, tickers=["ZZACM"], client=client)
        assert [r.accession for r, _ in done] == [acc]
        stats = index_raw_dir(cfg2, store, workers=2)
        assert stats.filings_indexed == 1 and stats.filings_failed == 0 and stats.chunks > 20
        assert index_raw_dir(cfg2, store, workers=2).filings_skipped == 1  # already current
        row = conn.execute(
            "SELECT ticker, fiscal_year, fiscal_period, synthetic, source_url "
            "FROM filings WHERE accession = %s",
            (acc,),
        ).fetchone()
        assert row[:4] == ("ZZACM", 2024, "FY", False)
        assert row[4] == (
            "https://www.sec.gov/Archives/edgar/data/9900001/000990000125999999/acme-20241231.htm"
        )
    finally:
        conn.execute("DELETE FROM filings WHERE accession = %s", (acc,))
        store.refresh_bm25_stats()
