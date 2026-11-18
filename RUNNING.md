# Running Filing Q&A end to end (macOS)

This walks from a fresh checkout to real, quotable numbers: 500+ filings indexed, a
human-verified golden set of 100+ questions, and an official eval run. Every command is
meant to be pasted into Terminal from the repository folder.

Times and costs below are rough estimates for planning, not measurements. The pipeline
prints the real token counts and dollar costs as it runs.

| stage | rough time | rough OpenAI cost |
|---|---|---|
| install + tests | 10 to 15 min | $0 |
| download ~600 filings | 15 to 40 min (mostly transfer, a few GB) | $0 |
| index (parse, chunk, BM25) | 15 to 30 min | $0 |
| embed (~40M tokens, text-embedding-3-small) | 30 to 60 min (rate limits) | about $1 |
| draft golden-set candidates | 5 to 10 min | about $0.30 |
| label 100+ items yourself | 2 to 4 hours, in as many sittings as you like | $0 |
| official eval (~120 questions) | 10 to 20 min | well under $1 |

## 0. Prerequisites

* macOS on Apple Silicon or Intel, 16 GB RAM recommended, about 15 GB free disk.
* [Homebrew](https://brew.sh), then:

  ```bash
  brew install python@3.12 git
  brew install --cask docker      # Docker Desktop; open it once so the engine starts
  ```

* An OpenAI API key with billing enabled.

## 1. Install

```bash
cd ~/code/filing-qa             # wherever the repository lives
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,local]"   # local = sentence-transformers + torch for the cross-encoder
```

## 2. Secrets in a local .env (gitignored)

```bash
cp .env.example .env
open -e .env
```

Fill in:

```
OPENAI_API_KEY=sk-...
SEC_USER_AGENT=Firstname Lastname you@yourdomain.com
FILINGQA_REVIEWER=your-initials
```

SEC requires a real name and contact email in the User-Agent; downloads refuse to start
without one. Leave `FILINGQA_DATABASE_URL` unset: each config file already names its own
database, and the override would apply to every config. Check that `.env` is ignored:

```bash
git check-ignore .env && echo "ok: .env is ignored"
```

## 3. Start Postgres + pgvector

```bash
docker compose up -d --wait
docker compose ps                                   # filingqa-db should be "healthy"
docker exec filingqa-db createdb -U filingqa filingqa_test
docker exec filingqa-db createdb -U filingqa filingqa_offline
```

The database listens on localhost:5433 (not 5432, to avoid clashing with a local Postgres).

## 4. Prove the install works

```bash
make lint
FILINGQA_TEST_DATABASE_URL=postgresql://filingqa:filingqa@localhost:5433/filingqa_test make test
make eval-offline
```

All tests should pass, and `make eval-offline` should end with
`gate (eval/thresholds.offline.json): PASS`. That run uses synthetic companies and
stand-in models: it proves the plumbing, its numbers mean nothing about quality.

## 5. Create the real schema and try a small slice first

```bash
make migrate
.venv/bin/python -m filingqa.ingest download --tickers AAPL,MSFT --max-filings 4
make index
make embed
make stats
```

Ask a question through the API (first start downloads the ~90 MB cross-encoder):

```bash
make serve          # leave running; use a second Terminal tab for the next command
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What were Apple'"'"'s total net sales in fiscal 2024?"}' | python3 -m json.tool
```

You should see an answer with citations (chunk id, section, source URL, quote) or a
refusal with a reason. Stop the server with Ctrl+C.

## 6. Ingest the full corpus (500+ filings)

```bash
make download       # ~100 companies, 10-K + 10-Q filed 2023-07-01 .. 2024-12-31
make index          # parses in 3-4 processes, loads chunks and BM25 postings
make embed          # prints tokens and cost_usd when done; builds the HNSW index
make stats
```

`make stats` must show `"filings"` of at least 500. If it is lower (a few tickers can fail
or file less often), widen the window in `config/default.yaml`, for example
`start_date: "2023-01-01"`, and run `make download index embed` again; already-downloaded
and already-indexed filings are skipped.

All three steps are resumable. If something is interrupted, run the same command again.
Failures are listed in the JSON line each step prints.

## 7. Draft golden-set candidates

```bash
make draft-golden
```

This writes about 140 answerable and 25 unanswerable candidates to
`eval/golden_drafts.jsonl` (gitignored). They are LLM drafts: unverified, and the eval
ignores them.

## 8. Label: you verify every item

```bash
make label
```

For each draft: `a` accept, `e` edit, `w` write your own question for the passage,
`r` reject, `s` skip, `q` quit. Every decision is saved immediately; run `make label`
again to continue where you stopped.

Guidelines that keep the set honest and useful:

* Accept only if the highlighted quote really supports the answer, and the question names
  the company and the period (fix it with `e` if not).
* Reject vague or trivial questions. Prefer a mix: numbers from tables, risk factors,
  MD&A explanations, 10-Q items.
* For unanswerable drafts, read the top retrieved passages shown; accept only if none of
  them answers the question.
* Writing some questions yourself with `w` is encouraged; the results record how many
  items were accepted unchanged, edited, or written by you.

Aim for 110 to 130 verified items including 15 to 25 unanswerable, so a few later
rejections do not drop you under the 100 minimum. Commit the golden set when done:

```bash
git add eval/golden.jsonl && git commit -m "Golden set: N human-verified items"
```

## 9. Run the official eval

```bash
make eval
```

It prints the verified count first, refuses to run below 100, compares all four retrieval
modes on the same questions, then runs end-to-end answers with the faithfulness judge.

Results land in:

* `results/latest.md`: the tables (retrieval per mode, end-to-end metrics, refusal
  threshold sweep)
* `results/latest.json` and `results/eval_<timestamp>.json`: everything, including every
  question's retrieved chunks, answer, citations, judge verdicts, latency and cost

## 10. Turn on the regression gate

```bash
make thresholds     # writes eval/thresholds.json from results/latest.json
make eval           # should now end with "gate (eval/thresholds.json): PASS"
git add eval/thresholds.json && git add -f results/latest.md
git commit -m "Baseline eval thresholds from the first official run"
```

From now on `make eval` exits 1 if a metric regresses past its bound.

## 11. Reading the numbers

| claim | where it comes from |
|---|---|
| filings indexed | `make stats` → `filings` |
| golden-set size | `results/latest.md` → human-verified items (and the provenance line) |
| recall@5 baseline → improved | `vector` row vs `hybrid_rerank` row, recall@5, same run |
| faithfulness | end-to-end table → faithfulness |
| p95 latency | end-to-end table → latency p95 (includes the LLM call) |
| $/query | end-to-end table → cost per query (embedding + answer LLM; judge excluded) |
| refusal behaviour | refusal accuracy and false refusal rate |

Copy them into the README results section exactly as measured, with the item counts. If a number is worse than you hoped, change the system (chunk size,
`candidates`, `rerank_candidates`, refusal threshold picked from the sweep), rerun
`make eval`, and report what that run measures.

Useful variations:

* Answer quality for the baseline too (about twice the answer and judge cost):
  `.venv/bin/python -m filingqa.eval --answer-modes vector,hybrid_rerank`
* After changing `chunking:` settings, re-chunk everything and re-embed (embeddings of
  replaced chunks are deleted automatically, so this repeats the ~$1 embedding cost):
  `.venv/bin/python -m filingqa.ingest index --force && make embed`
  Golden-set labels survive re-chunking: gold passages are character spans in the filing
  text, re-checked against the index at the start of every eval.

## 12. Optional: the live eval in GitHub Actions

In your GitHub repository: Settings → Secrets and variables → Actions → add
`OPENAI_API_KEY` and `SEC_USER_AGENT`. Then Actions → ci → Run workflow. The `live-eval`
job downloads only the filings your golden set cites, runs the eval on that smaller corpus
(numbers differ from the official run), and uploads results as an artifact. Baseline its
gate once:

```bash
python -m filingqa.eval.gate path/to/downloaded/latest.json eval/thresholds.ci.json --write
```

## Troubleshooting

* **`SEC returned 403`**: `SEC_USER_AGENT` is missing or generic, or you exceeded the rate
  limit. Use your real name and email, keep `max_requests_per_second` at 8 or lower.
* **OpenAI 429s**: retried automatically with backoff. If embedding is slow, that is your
  account's tokens-per-minute limit; let it run or lower `embedding.batch_size`.
* **`model ... not found`**: OpenAI retired the model; set `llm.model` and
  `eval.judge.model` in `config/default.yaml` and add its price to `config/prices.yaml`.
* **Port 5433 in use**: change the left side of `"5433:5432"` in `docker-compose.yml` and
  the URLs in `config/*.yaml`.
* **`REFUSING official eval ... synthetic test filing`**: the synthetic corpus got into your
  main database. Remove it: `docker exec filingqa-db psql -U filingqa -d filingqa -c
  "DELETE FROM filings WHERE synthetic"` then `make index` to refresh BM25 statistics.
* **Start over**: `docker compose down -v` deletes the database volume (raw downloads in
  `data/` are kept).
