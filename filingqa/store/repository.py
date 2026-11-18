"""All SQL lives here: loading filings/chunks, BM25 and vector search, and lookups."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import psycopg
from psycopg import sql

from filingqa.ingest.chunker import Chunk, FilingMeta
from filingqa.ingest.html_parser import ParsedFiling


@dataclass
class ChunkRecord:
    id: int
    chunk_uid: str
    accession: str
    company: str
    ticker: str | None
    cik: int
    form_type: str
    fiscal_year: int | None
    fiscal_period: str | None
    filing_date: date | None
    period_of_report: date | None
    section_id: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    context_header: str
    has_table: bool
    source_url: str

    @property
    def display_text(self) -> str:
        """What the reranker and the LLM see: header (company, period, section) + passage."""
        return f"{self.context_header}\n{self.text}"


@dataclass
class SearchFilters:
    tickers: list[str] | None = None
    form_types: list[str] | None = None
    fiscal_years: list[int] | None = None
    accessions: list[str] | None = None

    def is_empty(self) -> bool:
        return not (self.tickers or self.form_types or self.fiscal_years or self.accessions)


_CHUNK_COLUMNS = """
    c.id, c.chunk_uid, c.accession, f.company, f.ticker, f.cik, f.form_type, f.fiscal_year,
    f.fiscal_period, f.filing_date, f.period_of_report, c.section_id, c.section_title,
    c.chunk_index, c.char_start, c.char_end, c.text, c.context_header, c.has_table, f.source_url
"""


def vector_literal(v: Sequence[float] | np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


def index_name_for(model: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[:30]
    digest = hashlib.sha1(f"{model}:{dim}".encode()).hexdigest()[:8]
    return f"emb_hnsw_{slug}_{dim}_{digest}"


class ChunkStore:
    """Data access over a psycopg connection or a psycopg_pool.ConnectionPool."""

    def __init__(self, source: Any) -> None:
        self._source = source

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        if isinstance(self._source, psycopg.Connection):
            yield self._source
        else:
            with self._source.connection() as conn:
                yield conn

    # --- ingestion ------------------------------------------------------------------

    def filing_is_current(self, accession: str, parser_version: str, chunker_version: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT parser_version, chunker_version FROM filings WHERE accession = %s",
                (accession,),
            ).fetchone()
        return bool(row) and row[0] == parser_version and row[1] == chunker_version

    def replace_filing(
        self,
        meta: FilingMeta,
        parsed: ParsedFiling,
        chunks: list[Chunk],
        chunker_version: str,
    ) -> int:
        """Insert (or re-insert) one filing with its text, chunks and BM25 postings."""
        with self._conn() as conn, conn.transaction():
            conn.execute("DELETE FROM filings WHERE accession = %s", (meta.accession,))
            conn.execute(
                """INSERT INTO filings (accession, cik, ticker, company, form_type, filing_date,
                       period_of_report, fiscal_year, fiscal_period, source_url, parser_version,
                       chunker_version, n_chars, n_sections, synthetic)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    meta.accession,
                    meta.cik,
                    meta.ticker,
                    meta.company,
                    meta.form_type,
                    meta.filing_date,
                    meta.period_of_report,
                    meta.fiscal_year,
                    meta.fiscal_period,
                    meta.source_url,
                    parsed.parser_version,
                    chunker_version,
                    len(parsed.text),
                    len(parsed.sections),
                    meta.synthetic,
                ),
            )
            conn.execute(
                "INSERT INTO filing_texts (accession, text) VALUES (%s, %s)",
                (meta.accession, parsed.text),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (chunk_uid, accession, chunk_index, section_id,
                           section_title, section_chunk_index, char_start, char_end, text,
                           context_header, has_table)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    [
                        (
                            c.chunk_uid,
                            c.accession,
                            c.chunk_index,
                            c.section_id,
                            c.section_title,
                            c.section_chunk_index,
                            c.char_start,
                            c.char_end,
                            c.text,
                            c.context_header,
                            c.has_table,
                        )
                        for c in chunks
                    ],
                )
            # BM25 document length = number of indexed tokens (positions) in the tsvector.
            conn.execute(
                """UPDATE chunks c SET doc_len = (
                       SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 0)
                       FROM unnest(c.tsv) AS u)
                   WHERE c.accession = %s""",
                (meta.accession,),
            )
            conn.execute(
                """INSERT INTO chunk_terms (lexeme, chunk_id, tf, doc_len)
                   SELECT u.lexeme, c.id, coalesce(array_length(u.positions, 1), 1), c.doc_len
                   FROM chunks c CROSS JOIN LATERAL unnest(c.tsv) AS u
                   WHERE c.accession = %s""",
                (meta.accession,),
            )
        return len(chunks)

    def refresh_bm25_stats(self) -> None:
        with self._conn() as conn:
            conn.execute("REFRESH MATERIALIZED VIEW bm25_df")
            conn.execute("REFRESH MATERIALIZED VIEW bm25_corpus")
            # Fresh planner statistics: without them the planner misestimates the BM25 join.
            for table in ("bm25_df", "bm25_corpus", "chunk_terms", "chunks"):
                conn.execute(f"ANALYZE {table}")

    def bm25_stats_stale(self) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT (SELECT n_docs FROM bm25_corpus), (SELECT count(*) FROM chunks)"
            ).fetchone()
        return row is None or row[0] != row[1]

    # --- embeddings -----------------------------------------------------------------

    def chunks_missing_embeddings(self, model: str, limit: int) -> list[tuple[int, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.id, c.context_header || E'\\n' || c.text
                   FROM chunks c
                   WHERE NOT EXISTS (SELECT 1 FROM chunk_embeddings e
                                     WHERE e.chunk_id = c.id AND e.model = %s)
                   ORDER BY c.id LIMIT %s""",
                (model, limit),
            ).fetchall()
        return [(int(r[0]), r[1]) for r in rows]

    def count_missing_embeddings(self, model: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT count(*) FROM chunks c WHERE NOT EXISTS (
                       SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = c.id AND e.model = %s)""",
                (model,),
            ).fetchone()
        return int(row[0]) if row else 0

    def insert_embeddings(self, model: str, dim: int, rows: Sequence[tuple[int, np.ndarray]]) -> None:
        with (
            self._conn() as conn,
            conn.transaction(),
            conn.cursor() as cur,
            cur.copy("COPY chunk_embeddings (chunk_id, model, dim, embedding) FROM STDIN") as copy,
        ):
            for chunk_id, vec in rows:
                if len(vec) != dim:
                    raise ValueError(f"embedding has {len(vec)} dims, expected {dim}")
                copy.write_row((chunk_id, model, dim, vector_literal(vec)))

    def ensure_vector_index(self, model: str, dim: int) -> str:
        """Partial HNSW index (cosine) over this model's vectors, cast to a fixed dimension."""
        name = index_name_for(model, dim)
        stmt = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON chunk_embeddings "
            "USING hnsw ((embedding::vector({dim})) vector_cosine_ops) WHERE model = {model}"
        ).format(name=sql.Identifier(name), dim=sql.Literal(int(dim)), model=sql.Literal(model))
        with self._conn() as conn:
            conn.execute(stmt)
            conn.execute("ANALYZE chunk_embeddings")
        return name

    # --- search -----------------------------------------------------------------------

    @staticmethod
    def _filter_sql(filters: SearchFilters | None) -> tuple[sql.Composable, dict[str, Any]]:
        if filters is None or filters.is_empty():
            return sql.SQL(""), {}
        parts: list[sql.Composable] = []
        params: dict[str, Any] = {}
        if filters.tickers:
            parts.append(sql.SQL("f.ticker = ANY(%(f_tickers)s)"))
            params["f_tickers"] = [t.upper() for t in filters.tickers]
        if filters.form_types:
            parts.append(sql.SQL("f.form_type = ANY(%(f_forms)s)"))
            params["f_forms"] = [t.upper() for t in filters.form_types]
        if filters.fiscal_years:
            parts.append(sql.SQL("f.fiscal_year = ANY(%(f_years)s)"))
            params["f_years"] = list(filters.fiscal_years)
        if filters.accessions:
            parts.append(sql.SQL("f.accession = ANY(%(f_accs)s)"))
            params["f_accs"] = list(filters.accessions)
        return sql.SQL(" AND ") + sql.SQL(" AND ").join(parts), params

    def vector_search(
        self,
        model: str,
        dim: int,
        query_vec: Sequence[float] | np.ndarray,
        k: int,
        filters: SearchFilters | None = None,
        ef_search: int = 100,
    ) -> list[tuple[int, float]]:
        """Top-k chunk ids by cosine similarity (1 - cosine distance), best first."""
        where, params = self._filter_sql(filters)
        join = (
            sql.SQL(" JOIN chunks c ON c.id = e.chunk_id JOIN filings f ON f.accession = c.accession")
            if params
            else sql.SQL("")
        )
        dist = sql.SQL("e.embedding::vector({d}) <=> %(q)s::vector({d})").format(d=sql.Literal(int(dim)))
        stmt = sql.SQL(
            "SELECT e.chunk_id, 1 - ({dist}) AS sim FROM chunk_embeddings e{join} "
            "WHERE e.model = {model}{where} ORDER BY {dist} LIMIT %(k)s"
        ).format(dist=dist, join=join, model=sql.Literal(model), where=where)
        params.update({"q": vector_literal(query_vec), "k": int(k)})
        with self._conn() as conn, conn.transaction():
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(ef_search, k)),))
            rows = conn.execute(stmt, params).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def bm25_search(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[tuple[int, float]]:
        """Okapi BM25 over chunk_terms, using the same 'english' analyzer as the index.

        score(d) = sum over query terms t of
                   idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avglen))
        idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        """
        where, params = self._filter_sql(filters)
        join = (
            sql.SQL(" JOIN chunks c ON c.id = t.chunk_id JOIN filings f ON f.accession = c.accession")
            if params
            else sql.SQL("")
        )
        stmt = sql.SQL(
            """
            WITH q AS (
                SELECT DISTINCT lexeme
                FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS lexeme
            ),
            stats AS (SELECT n_docs::float8 AS n, greatest(avgdl, 1.0) AS avgdl FROM bm25_corpus),
            qt AS (
                SELECT q.lexeme, ln(1 + (s.n - d.df + 0.5) / (d.df + 0.5)) AS idf
                FROM q JOIN bm25_df d ON d.lexeme = q.lexeme CROSS JOIN stats s
            )
            SELECT t.chunk_id,
                   sum(qt.idf * (t.tf * (%(k1)s + 1.0))
                       / (t.tf + %(k1)s * (1.0 - %(b)s + %(b)s * t.doc_len / s.avgdl))) AS score
            FROM qt
            JOIN chunk_terms t ON t.lexeme = qt.lexeme
            CROSS JOIN stats s{join}
            WHERE true{where}
            GROUP BY t.chunk_id
            ORDER BY score DESC, t.chunk_id
            LIMIT %(k)s
            """
        ).format(join=join, where=where)
        params.update({"query": query, "k1": float(k1), "b": float(b), "k": int(k)})
        with self._conn() as conn:
            rows = conn.execute(stmt, params).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def vector_similarities(
        self, model: str, dim: int, query_vec: Sequence[float] | np.ndarray, ids: Sequence[int]
    ) -> dict[int, float]:
        if not ids:
            return {}
        stmt = sql.SQL(
            "SELECT chunk_id, 1 - (embedding::vector({d}) <=> %s::vector({d})) "
            "FROM chunk_embeddings WHERE model = {m} AND chunk_id = ANY(%s)"
        ).format(d=sql.Literal(int(dim)), m=sql.Literal(model))
        with self._conn() as conn:
            rows = conn.execute(stmt, (vector_literal(query_vec), list(ids))).fetchall()
        return {int(r[0]): float(r[1]) for r in rows}

    # --- lookups ----------------------------------------------------------------------

    def get_chunks(self, ids: Sequence[int]) -> dict[int, ChunkRecord]:
        if not ids:
            return {}
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN filings f ON f.accession = c.accession "
                "WHERE c.id = ANY(%s)",
                (list(ids),),
            ).fetchall()
        return {int(r[0]): ChunkRecord(*r) for r in rows}

    def get_chunk_by_uid(self, chunk_uid: str) -> ChunkRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN filings f ON f.accession = c.accession "
                "WHERE c.chunk_uid = %s",
                (chunk_uid,),
            ).fetchone()
        return ChunkRecord(*row) if row else None

    def filing_text(self, accession: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT text FROM filing_texts WHERE accession = %s", (accession,)).fetchone()
        return row[0] if row else None

    def sample_chunks(
        self,
        n: int,
        *,
        seed: float = 0.42,
        min_chars: int = 400,
        section_ids: Sequence[str] | None = None,
        include_synthetic: bool = False,
        has_table: bool | None = None,
        exclude_uids: Sequence[str] | None = None,
    ) -> list[ChunkRecord]:
        """Random sample of substantive chunks, spread across filings (for golden drafts)."""
        params: dict[str, Any] = {"n": int(n), "min_chars": min_chars, "seed": seed}
        extra = ""
        if section_ids:
            extra += " AND c.section_id = ANY(%(sections)s)"
            params["sections"] = list(section_ids)
        if not include_synthetic:
            extra += " AND NOT f.synthetic"
        if has_table is not None:
            extra += " AND c.has_table = %(has_table)s"
            params["has_table"] = has_table
        if exclude_uids:
            extra += " AND NOT (c.chunk_uid = ANY(%(exclude)s))"
            params["exclude"] = list(exclude_uids)
        with self._conn() as conn, conn.transaction():
            conn.execute("SELECT setseed(%(seed)s)", params)
            rows = conn.execute(
                f"""SELECT {_CHUNK_COLUMNS} FROM (
                        SELECT c.*, row_number() OVER (PARTITION BY c.accession ORDER BY random()) AS rn
                        FROM chunks c JOIN filings f ON f.accession = c.accession
                        WHERE length(c.text) >= %(min_chars)s{extra}
                    ) c JOIN filings f ON f.accession = c.accession
                    ORDER BY c.rn, random() LIMIT %(n)s""",
                params,
            ).fetchall()
        return [ChunkRecord(*r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            n_filings, n_companies = conn.execute(
                "SELECT count(*), count(DISTINCT cik) FROM filings WHERE NOT synthetic"
            ).fetchone()
            n_synth = conn.execute("SELECT count(*) FROM filings WHERE synthetic").fetchone()[0]
            by_form = dict(conn.execute("SELECT form_type, count(*) FROM filings GROUP BY 1").fetchall())
            n_chunks, n_table_chunks = conn.execute(
                "SELECT count(*), count(*) FILTER (WHERE has_table) FROM chunks"
            ).fetchone()
            emb = dict(conn.execute("SELECT model, count(*) FROM chunk_embeddings GROUP BY 1").fetchall())
        return {
            "filings": int(n_filings),
            "companies": int(n_companies),
            "synthetic_filings": int(n_synth),
            "filings_by_form": {k: int(v) for k, v in by_form.items()},
            "chunks": int(n_chunks),
            "table_chunks": int(n_table_chunks),
            "embeddings_by_model": {k: int(v) for k, v in emb.items()},
        }
