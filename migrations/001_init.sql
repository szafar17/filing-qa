-- Filing Q&A schema.
-- Applied in order by `python -m filingqa.store migrate`; each file runs once
-- and is recorded in schema_migrations.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per SEC filing (a 10-K or 10-Q primary document).
CREATE TABLE IF NOT EXISTS filings (
    accession        text PRIMARY KEY,          -- e.g. 0000320193-23-000106
    cik              bigint NOT NULL,
    ticker           text,
    company          text NOT NULL,
    form_type        text NOT NULL,             -- 10-K or 10-Q
    filing_date      date,
    period_of_report date,
    fiscal_year      int,
    fiscal_period    text,                      -- FY, Q1, Q2, Q3
    source_url       text NOT NULL,             -- primary document on sec.gov
    parser_version   text NOT NULL,
    chunker_version  text NOT NULL,
    n_chars          int NOT NULL,
    n_sections       int NOT NULL,
    synthetic        boolean NOT NULL DEFAULT false,  -- true only for test fixtures
    ingested_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS filings_ticker_idx ON filings (ticker);

-- Full normalized text of each filing. Chunk and citation character offsets
-- point into this text, which is what makes citations checkable.
CREATE TABLE IF NOT EXISTS filing_texts (
    accession text PRIMARY KEY REFERENCES filings (accession) ON DELETE CASCADE,
    text      text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id               bigserial PRIMARY KEY,
    chunk_uid        text NOT NULL UNIQUE,      -- "<accession>:<chunk_index>"
    accession        text NOT NULL REFERENCES filings (accession) ON DELETE CASCADE,
    chunk_index      int NOT NULL,              -- position within the filing
    section_id       text NOT NULL,             -- "7", "1A", "I-2", "II-1A", "cover"
    section_title    text NOT NULL,
    section_chunk_index int NOT NULL,           -- position within the section
    char_start       int NOT NULL,              -- offsets into filing_texts.text
    char_end         int NOT NULL,
    text             text NOT NULL,             -- exactly filing_texts.text[char_start:char_end]
    context_header   text NOT NULL,             -- company / form / period / section (+ table header)
    has_table        boolean NOT NULL DEFAULT false,
    -- Keyword index over header + passage. 'english' = stemming + stopwords.
    tsv              tsvector GENERATED ALWAYS AS
                       (to_tsvector('english', context_header || ' ' || text)) STORED,
    doc_len          int,                       -- token count used by BM25 length normalization
    CHECK (char_end > char_start)
);
CREATE INDEX IF NOT EXISTS chunks_accession_idx ON chunks (accession);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Inverted index for real Okapi BM25 (Postgres ts_rank is not BM25: it has no IDF).
-- One row per (term, chunk) with the term frequency, built from chunks.tsv.
CREATE TABLE IF NOT EXISTS chunk_terms (
    lexeme   text NOT NULL,
    chunk_id bigint NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    tf       int NOT NULL,
    doc_len  int NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_terms_lexeme_idx
    ON chunk_terms (lexeme) INCLUDE (chunk_id, tf, doc_len);
CREATE INDEX IF NOT EXISTS chunk_terms_chunk_idx ON chunk_terms (chunk_id);

-- Corpus statistics for BM25, refreshed after every ingest run.
CREATE MATERIALIZED VIEW IF NOT EXISTS bm25_df AS
    SELECT lexeme, count(*)::int AS df FROM chunk_terms GROUP BY lexeme;
CREATE UNIQUE INDEX IF NOT EXISTS bm25_df_lexeme_idx ON bm25_df (lexeme);

CREATE MATERIALIZED VIEW IF NOT EXISTS bm25_corpus AS
    SELECT count(*)::bigint AS n_docs,
           coalesce(avg(doc_len), 0)::float8 AS avgdl
    FROM chunks;

-- Embeddings live in their own table keyed by model so several embedding models
-- (OpenAI, local sentence-transformers, offline hashing) can index the same chunks.
-- The column is dimension-less; each model gets a partial HNSW expression index
-- with its own dimension, created by the indexer (see store/repository.py).
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id  bigint NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    model     text NOT NULL,
    dim       int NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
