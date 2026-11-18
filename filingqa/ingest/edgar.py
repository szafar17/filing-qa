"""SEC EDGAR client: ticker -> CIK, filing lists, and primary-document downloads.

Fair-access rules (https://www.sec.gov/os/accessing-edgar-data) are enforced in code:
  * a descriptive User-Agent with contact info is required (read from SEC_USER_AGENT;
    we refuse to run without it),
  * at most 10 requests/second (default 8) via a shared, thread-safe rate limiter,
  * exponential backoff with jitter on 429/5xx and network errors, honoring Retry-After.

Downloads are cached under data/raw/<cik>/<accession>/ with a filing.json sidecar, so
re-running is idempotent and resumable.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import httpx

from filingqa.retry import backoff_delay, parse_retry_after

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

RETRY_STATUS = {429, 500, 502, 503, 504}


class UserAgentError(RuntimeError):
    pass


def get_user_agent(env: dict[str, str] | None = None) -> str:
    """Return SEC_USER_AGENT or raise: SEC blocks clients without contact info."""
    env = dict(os.environ) if env is None else env
    ua = (env.get("SEC_USER_AGENT") or "").strip()
    if not ua:
        raise UserAgentError(
            "SEC_USER_AGENT is not set. SEC's fair-access policy requires a User-Agent that "
            "identifies you, e.g. 'Jane Doe jane.doe@yourdomain.com'. Put it in your .env file."
        )
    if "@" not in ua or " " not in ua:
        raise UserAgentError(
            "SEC_USER_AGENT must contain a name and a contact email, e.g. 'Jane Doe jane.doe@yourdomain.com'."
        )
    if "example.com" in ua.lower():
        raise UserAgentError("SEC_USER_AGENT still contains the example address; use your own.")
    return ua


class RateLimiter:
    """Thread-safe limiter: successive acquire() calls are spaced >= 1/rate seconds apart."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self.interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next - now
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
            self._next = max(now, self._next) + self.interval


@dataclass
class FilingRef:
    cik: int
    ticker: str | None
    company: str
    form_type: str
    accession: str
    filing_date: date
    period_of_report: date | None
    primary_document: str
    fiscal_year_end: str | None = None  # MMDD, from the submissions API

    @property
    def acc_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc_nodash=self.acc_nodash, doc=self.primary_document)

    def to_json(self) -> dict:
        d = asdict(self)
        d["filing_date"] = self.filing_date.isoformat()
        d["period_of_report"] = self.period_of_report.isoformat() if self.period_of_report else None
        d["source_url"] = self.url
        return d

    @classmethod
    def from_json(cls, d: dict) -> FilingRef:
        return cls(
            cik=int(d["cik"]),
            ticker=d.get("ticker"),
            company=d["company"],
            form_type=d["form_type"],
            accession=d["accession"],
            filing_date=date.fromisoformat(d["filing_date"]),
            period_of_report=(
                date.fromisoformat(d["period_of_report"]) if d.get("period_of_report") else None
            ),
            primary_document=d["primary_document"],
            fiscal_year_end=d.get("fiscal_year_end"),
        )


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _rows_from_columns(cols: dict) -> Iterable[dict]:
    """The submissions API stores filings column-wise; yield them row-wise."""
    keys = list(cols)
    n = len(cols.get("accessionNumber", []))
    for i in range(n):
        yield {k: (cols[k][i] if i < len(cols[k]) else None) for k in keys}


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        *,
        max_requests_per_second: float = 8.0,
        max_attempts: int = 6,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        limiter: RateLimiter | None = None,
        timeout: float = 30.0,
    ) -> None:
        if max_requests_per_second > 10:
            raise ValueError("SEC fair-access limit is 10 requests/second")
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._limiter = limiter or RateLimiter(max_requests_per_second, sleep=sleep)
        self._sleep = sleep
        self._max_attempts = max_attempts
        self.requests_made = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, url: str) -> httpx.Response:
        attempt = 0
        while True:
            attempt += 1
            self._limiter.acquire()
            self.requests_made += 1
            try:
                resp = self._client.get(url)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= self._max_attempts:
                    raise
                delay = backoff_delay(attempt, 1.0, 60.0, 0.25)
                log.warning(
                    "sec request failed, retrying",
                    extra={
                        "data": {
                            "event": "sec_retry",
                            "url": url,
                            "error": type(exc).__name__,
                            "attempt": attempt,
                            "delay_s": round(delay, 2),
                        }
                    },
                )
                self._sleep(delay)
                continue
            if resp.status_code in RETRY_STATUS and attempt < self._max_attempts:
                delay = parse_retry_after(resp.headers.get("Retry-After"))
                if delay is None:
                    delay = backoff_delay(attempt, 1.0, 60.0, 0.25)
                log.warning(
                    "sec throttled or errored, backing off",
                    extra={
                        "data": {
                            "event": "sec_retry",
                            "url": url,
                            "status": resp.status_code,
                            "attempt": attempt,
                            "delay_s": round(delay, 2),
                        }
                    },
                )
                self._sleep(min(delay, 120.0))
                continue
            if resp.status_code == 403:
                raise PermissionError(
                    f"SEC returned 403 for {url}. Check SEC_USER_AGENT and that you stay under "
                    "10 requests/second; SEC may block clients that do not."
                )
            resp.raise_for_status()
            return resp

    # --- metadata -----------------------------------------------------------------

    def ticker_map(self) -> dict[str, tuple[int, str]]:
        data = self.get(TICKERS_URL).json()
        out: dict[str, tuple[int, str]] = {}
        for row in data.values():
            out[str(row["ticker"]).upper()] = (int(row["cik_str"]), str(row["title"]))
        return out

    def list_filings(
        self,
        cik: int,
        *,
        forms: Iterable[str],
        start: date,
        end: date,
        ticker: str | None = None,
        include_amendments: bool = False,
    ) -> list[FilingRef]:
        """Filings of the given forms with filing dates in [start, end], newest first."""
        wanted = {f.upper() for f in forms}
        if include_amendments:
            wanted |= {f + "/A" for f in wanted}
        sub = self.get(SUBMISSIONS_URL.format(cik=cik)).json()
        company = sub.get("name") or str(cik)
        fye = sub.get("fiscalYearEnd")
        pages = [sub.get("filings", {}).get("recent", {})]
        # Large filers (banks issuing many prospectuses) push older filings into extra pages.
        for f in sub.get("filings", {}).get("files", []) or []:
            f_from, f_to = _parse_date(f.get("filingFrom")), _parse_date(f.get("filingTo"))
            if f_from and f_to and (f_to < start or f_from > end):
                continue
            pages.append(self.get(SUBMISSIONS_PAGE_URL.format(name=f["name"])).json())
        refs: dict[str, FilingRef] = {}
        for cols in pages:
            for row in _rows_from_columns(cols):
                form = str(row.get("form") or "").upper()
                fdate = _parse_date(row.get("filingDate"))
                if form not in wanted or fdate is None or not (start <= fdate <= end):
                    continue
                doc = row.get("primaryDocument")
                if not doc:
                    continue
                acc = row["accessionNumber"]
                refs[acc] = FilingRef(
                    cik=cik,
                    ticker=ticker,
                    company=company,
                    form_type=form,
                    accession=acc,
                    filing_date=fdate,
                    period_of_report=_parse_date(row.get("reportDate")),
                    primary_document=doc,
                    fiscal_year_end=fye,
                )
        return sorted(refs.values(), key=lambda r: r.filing_date, reverse=True)

    # --- downloads ----------------------------------------------------------------

    def download(self, ref: FilingRef, raw_dir: str | Path) -> Path:
        """Download the primary document (skips if already cached). Returns its path."""
        folder = Path(raw_dir) / str(ref.cik) / ref.accession
        doc_path = folder / ref.primary_document
        meta_path = folder / "filing.json"
        if doc_path.is_file() and doc_path.stat().st_size > 0 and meta_path.is_file():
            return doc_path
        folder.mkdir(parents=True, exist_ok=True)
        resp = self.get(ref.url)
        tmp = doc_path.with_suffix(doc_path.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(doc_path)
        meta_path.write_text(json.dumps(ref.to_json(), indent=2))
        return doc_path


def load_cached_refs(raw_dir: str | Path) -> list[tuple[FilingRef, Path]]:
    """All filings downloaded so far: (FilingRef, path to primary document)."""
    out: list[tuple[FilingRef, Path]] = []
    for meta_path in sorted(Path(raw_dir).glob("*/*/filing.json")):
        ref = FilingRef.from_json(json.loads(meta_path.read_text()))
        doc = meta_path.parent / ref.primary_document
        if doc.is_file():
            out.append((ref, doc))
    return out
