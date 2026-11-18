# Filing Q&A: Cited RAG over SEC Filings

Ask analyst questions about SEC 10-K and 10-Q filings and get answers that cite the exact
source passage. The system downloads filings from EDGAR, splits them into Items
(Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements, ...), turns financial
tables into readable row text, and indexes section-aware chunks in PostgreSQL for both
keyword search (Okapi BM25) and vector search (pgvector). A cross-encoder reranks the
fused candidates, an LLM answers from numbered passages in a strict JSON schema, and every
citation is checked against the source text before it is returned. When the evidence is
weak the service refuses instead of guessing.

An eval harness measures all of this on a human-verified golden set: recall@k and MRR per
retrieval mode, answer faithfulness (LLM-as-judge with a written rubric), refusal accuracy,
p50/p95 latency and dollars per query. A regression gate fails the build when a metric
drops below its committed threshold.

**Stack:** Python, FastAPI, PostgreSQL + pgvector, BM25, OpenAI API, sentence-transformers
cross-encoders, pytest, GitHub Actions.

> **Status.** The code, tests and CI are complete and pass. The real numbers (recall, faithfulness,
> latency, cost) come from running it on your machine against the live EDGAR corpus and the
> OpenAI API, which this repository has not been run against yet. See
> [RUN_ON_MAC.md](RUN_ON_MAC.md). The results table below stays empty until then.

## Architecture

```mermaid
flowchart LR
  subgraph ingest["Ingestion (python -m filingqa.ingest)"]
    edgar["EDGAR APIs<br/>rate limited, SEC_USER_AGENT"] --> raw["data/raw<br/>10-K / 10-Q HTML + filing.json"]
    raw --> parse["Parser<br/>Items, table rows, char offsets"]
    parse --> chunk["Section-aware chunker<br/>metadata + context header"]
  end
  chunk --> pg[("PostgreSQL<br/>filings, filing_texts, chunks")]
  pg --> bm25["tsvector + chunk_terms<br/>Okapi BM25 in SQL"]
  chunk --> emb["Embedder<br/>OpenAI / local / hashing"]
  emb --> vec[("pgvector<br/>HNSW index per model")]
  subgraph query["Answering (POST /ask)"]
    q["question"] --> kw["BM25 top 50"]
    q --> vs["vector top 50"]
    kw --> rrf["Reciprocal Rank Fusion"]
    vs --> rrf
    rrf --> ce["cross-encoder rerank<br/>top 30 to top 5"]
    ce --> gate{"evidence above<br/>threshold?"}
    gate -- no --> refuse["refusal + reason"]
    gate -- yes --> llm["LLM, numbered passages<br/>strict JSON schema"]
    llm --> check["citation check<br/>quote must occur in passage"]
    check --> resp["answer + citations<br/>+ JSON trace log"]
  end
  bm25 --> kw
  vec --> vs
  subgraph evalh["Eval (make eval)"]
    golden["golden set<br/>human-verified items"] --> run["all retrieval modes +<br/>end-to-end answers"]
    run --> results["results/*.json, latest.md"]
    results --> th{"thresholds.json<br/>regression gate"}
  end
```

## Quick start

The fastest way to see everything work needs no API key: the synthetic mini corpus with
offline stand-in components (what CI runs).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d --wait
docker exec filingqa-db createdb -U filingqa filingqa_offline
make eval-offline          # index 3 synthetic filings, run the smoke eval and its gate
```

For the real thing (500+ filings, OpenAI models, your own golden set) follow
[RUN_ON_MAC.md](RUN_ON_MAC.md) step by step.

## How it works

### 1. Ingestion (`filingqa/ingest/edgar.py`, `pipeline.py`)

* Tickers from `config/companies.yaml` (about 100 large US companies) are resolved to CIKs
  with SEC's `company_tickers.json`. The submissions API lists each company's filings;
  large filers (banks that issue thousands of prospectuses) push older filings into extra
  pages, which are fetched only when their date range overlaps the configured window.
* The default window (filed 2023-07-01 to 2024-12-31, 10-K and 10-Q, no amendments) gives
  roughly 600 filings.
* SEC fair-access rules are enforced in code: `SEC_USER_AGENT` must contain a name and an
  email or downloads refuse to start; a shared thread-safe limiter keeps the client at or
  below 8 requests per second (10 is SEC's ceiling and config validation rejects more);
  429 and 5xx responses back off exponentially with jitter and honour `Retry-After`.
* Downloads are cached under `data/raw/<cik>/<accession>/` with a `filing.json` sidecar,
  so re-runs are idempotent and resumable. Indexing parses in a process pool.

### 2. Parsing (`html_parser.py`, `sections.py`, `tables.py`)

* **Linearization.** The DOM is walked in document order (iteratively, since some filings
  nest thousands of tags) into text blocks. Inline-XBRL's hidden `ix:header` and anything
  with `display:none` is dropped; DEI facts (fiscal year, fiscal period, ticker) are read
  from it first. Running page headers and footers ("Acme Corp | 2024 Form 10-K | 17") and
  bare page numbers are removed.
* **Section detection.** "Item 7." headings appear in the table of contents, in the body,
  and in cross-references. Candidates are short blocks that start with "Item N[A-C]"
  (TOC links and page-numbered rows are excluded). A small dynamic program picks the
  subsequence whose items follow the canonical order (1, 1A, 1B, 1C, 2, ... 16 for a 10-K;
  Part I 1-4 then Part II 1-6 for a 10-Q) and that maximizes the sum of log(1 + section
  length / 200). The log is concave, so a TOC cluster of 50-character "sections" scores far
  below the real body headings, and a 10-Q's two "Item 1" headings resolve to Part I and
  Part II by order. If fewer than three Items are found (some banks file a narrative annual
  report with a cross-reference index) it falls back to headings that match canonical titles.
* **Tables.** SEC tables put "$" in its own cell, split negatives into "(1,234" and ")",
  span headers across columns and pad with empty spacer columns. The flattener merges
  those fragments, groups each value's physical columns into logical columns, matches
  multi-row headers to them, and emits one line per row:

  ```
  Columns: Three Months Ended June 30, 2024 | Three Months Ended June 30, 2023 | ...
  Total revenues | Three Months Ended June 30, 2024: $186.4 | Three Months Ended June 30, 2023: $135.9 | ...
  ```

  Tables without real numeric rows are layout tables and become plain text (this is how
  headings laid out in tables are found).
* **Offsets.** The full normalized text of each filing is stored in `filing_texts`. Every
  block, chunk and citation carries character offsets into it, so any quote can be checked.

### 3. Chunking (`chunker.py`)

* Chunks never cross an Item boundary. Prose is packed sentence by sentence up to
  1,600 characters (hard cap 2,400), with up to 200 characters of whole-sentence overlap.
* Table rows are never split. A chunk that continues a table gets the table's title and
  its `Columns:` line in its context header, so a row like `Diluted | 2024: $2.31` still
  says what it is.
* The context header (company, ticker, form, fiscal period, period end, filing date,
  section, table header) is embedded and keyword-indexed with the passage but stored
  separately, so `chunk.text == filing_text[char_start:char_end]` always holds.
* Metadata per chunk: company, CIK, ticker, form type, fiscal year and period, filing
  date, period of report, section id and title, chunk index (in filing and in section),
  source URL, character offsets, has-table flag.

### 4. Keyword search: real BM25 inside Postgres (`store/repository.py`, `migrations/001_init.sql`)

Postgres `ts_rank` is not BM25 (no inverse document frequency, different length
normalization). Here each chunk gets a stored `tsvector` (english stemming and stopwords,
GIN-indexed); at load time its terms are expanded into a `chunk_terms(lexeme, chunk_id, tf,
doc_len)` postings table with a covering index, and two materialized views hold document
frequencies and corpus size / average length. A query is analyzed with the same
`to_tsvector('english', ...)` and scored with Okapi BM25 (k1 = 1.2, b = 0.75) in one SQL
statement. A test recomputes BM25 in Python from the raw tsvectors and checks that the SQL
ranking and scores match exactly. JIT is disabled per connection: on these short index
lookups it added about 0.4 s of compile time per query.

### 5. Vector search (pgvector)

Embeddings live in `chunk_embeddings(chunk_id, model, dim, embedding vector)`, keyed by
model, so OpenAI, local and test embeddings can coexist. Each model gets its own partial
HNSW expression index, `USING hnsw ((embedding::vector(1536)) vector_cosine_ops) WHERE
model = '...'`. Vectors are L2-normalized; similarity is 1 minus cosine distance.

### 6. Retrieval modes (`retrieval/`)

| mode | what it does |
|---|---|
| `vector` | pgvector cosine top-k (the baseline) |
| `bm25` | Okapi BM25 top-k |
| `hybrid` | top 50 from each, fused with Reciprocal Rank Fusion: score = sum of 1 / (60 + rank) |
| `hybrid_rerank` | hybrid top 30 rescored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep top 5 |

RRF needs only ranks, so BM25 scores and cosine similarities need no normalization. The
cross-encoder reads query and passage together, which is more accurate than comparing two
independent embeddings but too slow for the whole corpus, hence retrieve first, then rerank.
Its logits are mapped to probabilities with a sigmoid so a refusal threshold has a stable
meaning. The mode is a config value and a per-request parameter, so the eval compares all
four on the same questions in one run.

### 7. Answering (`answer/`, `api/app.py`)

* `POST /ask {question, mode?, top_k?, filters?}` retrieves, then calls the LLM with the
  passages numbered `[1]..[5]` and a strict JSON schema (OpenAI Structured Outputs):
  `refused, refusal_reason, answer, confidence, citations[{passage, quote}]`.
* The output is validated with pydantic. Each citation's quote must occur in the cited
  passage's text (whitespace, case, quote-style tolerant); unverifiable citations are
  dropped, and the response carries the chunk id, company, form, fiscal period, section,
  source URL, the quoted span and its offsets in the filing.
* **Refusal path, three stages:** (1) before the LLM, if the best reranker probability
  (or cosine similarity when no reranker runs) is under the configured threshold;
  (2) the model abstains when the passages do not contain the answer; (3) an answer whose
  citations cannot be found in the passages becomes a refusal. Invalid structured output is
  retried once, then refused.
* OpenAI calls retry with exponential backoff and jitter on rate limits, timeouts and 5xx,
  honouring `Retry-After`; other errors surface immediately.
* Every request logs one JSON line: trace id, per-stage latency (embed, vector, BM25,
  fusion, rerank, LLM), tokens, cost, refusal stage, citation counts.

All model-facing parts sit behind small interfaces with offline doubles
(`HashingEmbedder`, `LexicalReranker`, `FakeLLM`), so tests and CI run with no API key and
no model downloads. The doubles are test fixtures, never used for reported numbers: the
official eval refuses to run with them.

## Evaluation

### Golden set and labeling (`eval/golden.py`, `draft.py`, `label.py`)

Items live in `eval/golden.jsonl`: question, `answerable`, gold passages (accession, quote,
character offsets, chunk id, source URL), reference answer, `labeled_by`, `verified`,
`verified_at`, `reviewer`, and provenance (`origin`, `edited`).

1. `make draft-golden` samples chunks across filings from the real index (about 30% table
   chunks) and asks the LLM for one analyst question per chunk, a reference answer and a
   verbatim supporting quote; the quote must be found in the chunk or the draft is dropped.
   It also drafts unanswerable questions for the refusal metric. Drafts go to
   `eval/golden_drafts.jsonl` with `labeled_by="llm-draft"`, `verified=false`.
2. `make label` is a terminal review loop. For each draft you see the question, the draft
   answer and the quote highlighted inside its passage (for unanswerable drafts, the top
   retrieved passages so you can confirm none answers it). Accept, edit (question, answer,
   quote, or mark unanswerable), write your own question for the passage, reject, or skip.
   Every decision is saved immediately; quitting and resuming is safe.
3. Only items a person accepted get `labeled_by="human"`, `verified=true` and a
   `verified_at` timestamp. Pydantic validation forbids verified drafts and verified
   synthetic fixtures. The official eval uses only verified items, refuses to run with
   fewer than 100, prints the verified count on every run, and records provenance (accepted
   unchanged, edited, hand-written) in the results.

The synthetic mini set (`tests/fixtures/mini_golden_SYNTHETIC.jsonl`, 16 items about three
invented companies) is test data for CI. It is marked `synthetic`, can never be verified,
and the official eval refuses to run against an index that contains synthetic filings.

### Metrics (`eval/metrics.py`, `eval/runner.py`)

* **Relevance.** A retrieved chunk is relevant to a gold passage if it comes from the same
  filing and covers at least 50% of the gold span. Gold spans are character ranges in the
  filing text; at the start of each run they are checked against the current index and
  relocated by their quote if the parser changed. Items whose gold passage cannot be found
  are skipped and reported, never counted.
* **recall@k** = gold passages found in the top k / gold passages (mean over answerable
  items). Drafted items have one gold passage, so recall@k equals hit@k; both are reported.
* **MRR** = mean of 1 / rank of the first relevant chunk (0 when none is retrieved).
* **Faithfulness** (LLM-as-judge): mean over answered questions of supported claims /
  total claims, plus the share of answers with every claim supported. Judge rubric:
  1. split the answer into atomic factual claims (a number, a date, a name, a direction
     of change, a cause), ignoring citation markers;
  2. a claim is supported only if the cited passages state it explicitly or it follows by
     trivial arithmetic or rewording;
  3. it is unsupported if absent, contradicted, about a different period, segment or
     company, or if it adds specifics the passages do not contain;
  4. judge only against the passages shown, never outside knowledge.
  The judge sees only the passages the answer cites. Caveat: the default judge is the same
  model family as the answerer, which can be lenient; set `eval.judge.model` to a stronger
  model for a stricter check.
* **Refusal accuracy** = unanswerable items refused / unanswerable items. **False refusal
  rate** = answerable items refused / answerable items. A threshold sweep in the report
  shows both rates for evidence thresholds from 0 to 0.95 (pre-LLM check only). Choose a
  threshold from the sweep, but with ~100 items, treat it as a calibration, not a result.
* **Citation overlap**: share of answered, answerable items with a citation that overlaps
  the gold span (a citation to another valid passage counts as a miss, so this is strict).
* **Schema-valid rate**: share of LLM answers valid on the first try.
* **Latency**: end-to-end wall time of `AnswerService.ask` (query embedding call, both
  searches, fusion, rerank, LLM call, validation), p50 and p95 by linear interpolation.
  Retrieval-only latency per mode is reported separately.
* **Cost per query** = query-embedding tokens x embedding price + answer input tokens x
  input price (cached input at the cached rate) + output tokens x output price. Token counts
  come from the API's `usage` field on each call; prices from `config/prices.yaml`, which
  records its `as_of` date and source. The local cross-encoder costs $0. Judge calls are
  eval overhead and are reported separately, not in $/query. Models missing from the price
  table are listed in the results instead of being priced at zero silently.

### Cost estimate method (before you run)

* Indexing: total chunk tokens x $0.02 / 1M for text-embedding-3-small. `make embed` prints
  the exact token count and cost from the API's usage numbers. As rough arithmetic, 600
  filings x roughly 60k tokens is roughly 36M tokens, about $0.70.
* Per query, illustrative arithmetic only: 5 passages of ~450 tokens plus ~350 tokens of
  instructions and question is ~2,600 input tokens; with ~150 output tokens on gpt-4.1-mini
  ($0.40 / $1.60 per 1M) that is about $0.0013. The eval measures the real value.

### Regression gate and CI

* `eval/thresholds.json` (official eval) is empty on purpose: thresholds must come from
  measured runs, never targets. After the first official run, `make thresholds` writes
  min/max bounds from `results/latest.json` (quality may drop 0.03, latency and cost may
  rise 25%) and `make eval` then exits 1 on any regression.
* CI (`.github/workflows/ci.yml`): ruff, pytest (unit plus integration tests against a
  `pgvector/pgvector:pg16` service), and an offline eval on the synthetic mini corpus gated by
  `eval/thresholds.offline.json`, which was baselined from a measured offline run. Breaking
  hybrid fusion, for example, fails the build. An optional `live-eval` job (manual or weekly,
  only when `OPENAI_API_KEY` and `SEC_USER_AGENT` secrets exist) downloads the filings the
  golden set cites, runs the real eval, and gates on `eval/thresholds.ci.json`.

## Results

Fill these in from real runs (`results/latest.md`). Nothing below has been measured yet.

| retrieval mode | recall@1 | recall@5 | recall@10 | MRR | p95 retrieval ms |
|---|---|---|---|---|---|
| vector (baseline) | fill in from real runs | | | | |
| bm25 | fill in from real runs | | | | |
| hybrid (RRF) | fill in from real runs | | | | |
| hybrid + cross-encoder | fill in from real runs | | | | |

| end-to-end (hybrid + cross-encoder) | value |
|---|---|
| golden set (human-verified items, of which unanswerable) | fill in from real runs |
| faithfulness | fill in from real runs |
| refusal accuracy / false refusal rate | fill in from real runs |
| p50 / p95 latency | fill in from real runs |
| cost per query | fill in from real runs |
| corpus (filings, chunks) | fill in from real runs |

## Configuration

`config/default.yaml` (OpenAI embeddings and LLM, local cross-encoder), `config/local.yaml`
(local bge-small embeddings), `config/offline.yaml` (tests/CI stand-ins), `config/prices.yaml`
(token prices), `config/companies.yaml` (tickers). Secrets come only from the environment
or a gitignored `.env` (see `.env.example`).

## Repository layout

```
filingqa/
  ingest/     edgar.py (downloader), html_parser.py, sections.py, tables.py, chunker.py,
              fiscal.py, pipeline.py, __main__.py (CLI)
  store/      db.py (connections, migrations), repository.py (all SQL: BM25, vectors, lookups)
  embeddings/ openai_embedder.py, local.py (sentence-transformers), hashing.py (offline)
  llm/        openai_llm.py (structured outputs + retries), fake.py (offline double)
  retrieval/  fusion.py (RRF), rerank.py (cross-encoder, lexical double), retriever.py
  answer/     prompts.py, schema.py (pydantic), service.py (refusal + citation checks)
  api/        app.py (FastAPI)
  eval/       golden.py, draft.py, label.py, metrics.py, faithfulness.py, runner.py,
              report.py, gate.py, __main__.py (make eval)
migrations/   001_init.sql
config/       default / local / offline configs, prices, companies
eval/         golden.jsonl (yours), thresholds*.json
tests/        unit + integration tests; fixtures/corpus/SYNTHETIC_*.html (invented companies)
```

## Tests

```bash
make test-unit                                   # no database needed
docker exec filingqa-db createdb -U filingqa filingqa_test
FILINGQA_TEST_DATABASE_URL=postgresql://filingqa:filingqa@localhost:5433/filingqa_test make test
```

The integration fixture wipes and recreates the schema, so it only accepts a database
whose name contains `test`.

## Limitations

* Only the primary document is indexed. A few issuers (Wells Fargo, for example) put most of
  the annual report in Exhibit 13 and incorporate it by reference, so their 10-K text is thin.
* Section detection is heuristic. Bank-style narrative reports fall back to title matching,
  and some sections may land in "cover" or a neighbouring Item.
* Fiscal labels come from inline-XBRL DEI facts when present; the date-based fallback can be
  off by one year for retailers that name a fiscal year by its start.
* Each drafted question has a single gold passage, so a correct answer from a different
  passage (the same number in MD&A and in Item 8) counts as a retrieval miss.
* With about 100 questions, p95 latency rests on the slowest handful of requests and every
  item moves a rate by about one point; report the item count with every number.

## Data notice

Filings come from SEC EDGAR and are public. The three HTML files under
`tests/fixtures/corpus/` are synthetic: invented companies (ZZ-prefixed tickers, fake CIKs,
`example.invalid` URLs) written to exercise the parser, and are not SEC filings.
