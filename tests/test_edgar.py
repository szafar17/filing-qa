"""EDGAR client tests. Everything runs against httpx.MockTransport: no network."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from filingqa.ingest.companies import load_tickers, normalize_ticker, resolve_companies
from filingqa.ingest.edgar import (
    EdgarClient,
    FilingRef,
    RateLimiter,
    UserAgentError,
    get_user_agent,
    load_cached_refs,
)

UA = "Test Runner tests@filingqa.invalid"


def test_user_agent_is_required():
    with pytest.raises(UserAgentError, match="not set"):
        get_user_agent({})
    with pytest.raises(UserAgentError):
        get_user_agent({"SEC_USER_AGENT": "no-email-here"})
    with pytest.raises(UserAgentError, match="example"):
        get_user_agent({"SEC_USER_AGENT": "Jane Doe jane@example.com"})
    with pytest.raises(UserAgentError, match="example"):
        get_user_agent({"SEC_USER_AGENT": "Firstname Lastname you@yourdomain.com"})
    assert get_user_agent({"SEC_USER_AGENT": f"  {UA} "}) == UA


class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def test_rate_limiter_spaces_requests():
    clock = FakeClock()
    rl = RateLimiter(8, clock=clock.now, sleep=clock.sleep)
    for _ in range(9):
        rl.acquire()
    # 9 requests at 8/s: the first is immediate, the rest wait 1/8 s each
    assert sum(clock.sleeps) == pytest.approx(8 / 8)
    assert clock.t == pytest.approx(101.0)


def test_client_rejects_rates_above_sec_limit():
    with pytest.raises(ValueError):
        EdgarClient(UA, max_requests_per_second=11)


def make_client(handler, sleeps: list[float] | None = None) -> EdgarClient:
    sleeps = sleeps if sleeps is not None else []
    clock = FakeClock()
    return EdgarClient(
        UA,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        limiter=RateLimiter(10, clock=clock.now, sleep=clock.sleep),
    )


def test_sends_user_agent_and_retries_429_with_retry_after():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        if len(seen) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    sleeps: list[float] = []
    client = make_client(handler, sleeps)
    assert client.get("https://data.sec.gov/x").json() == {"ok": True}
    assert seen == [UA, UA, UA]
    assert sleeps[0] == 3.0  # honoured Retry-After
    assert 1.0 <= sleeps[1] <= 3.0  # exponential backoff (2 s +/- jitter) for the 503


def test_retries_network_errors_then_raises():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("boom")

    client = make_client(handler)
    with pytest.raises(httpx.ConnectError):
        client.get("https://www.sec.gov/x")
    assert len(calls) == 6


def test_403_explains_fair_access():
    client = make_client(lambda r: httpx.Response(403))
    with pytest.raises(PermissionError, match="SEC_USER_AGENT"):
        client.get("https://www.sec.gov/x")


def submissions_payload():
    return {
        "name": "EXAMPLE CORP",
        "fiscalYearEnd": "0930",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000000001-24-000010",
                    "0000000001-24-000009",
                    "0000000001-24-000008",
                    "0000000001-23-000005",
                ],
                "form": ["10-K", "8-K", "10-Q/A", "10-Q"],
                "filingDate": ["2024-11-01", "2024-10-01", "2024-08-01", "2023-08-01"],
                "reportDate": ["2024-09-28", "", "2024-06-29", "2023-07-01"],
                "primaryDocument": ["exm-20240928.htm", "x.htm", "q.htm", "exm-20230701.htm"],
            },
            "files": [
                {
                    "name": "CIK0000000001-submissions-001.json",
                    "filingFrom": "2015-01-01",
                    "filingTo": "2022-12-31",
                },
                {
                    "name": "CIK0000000001-submissions-002.json",
                    "filingFrom": "2023-01-01",
                    "filingTo": "2023-06-30",
                },
            ],
        },
    }


def test_list_filings_filters_forms_dates_and_reads_older_pages():
    requested = []

    def handler(request):
        url = str(request.url)
        requested.append(url)
        if url.endswith("CIK0000000001.json"):
            return httpx.Response(200, json=submissions_payload())
        if url.endswith("submissions-002.json"):
            return httpx.Response(
                200,
                json={
                    "accessionNumber": ["0000000001-23-000002"],
                    "form": ["10-Q"],
                    "filingDate": ["2023-05-01"],
                    "reportDate": ["2023-04-01"],
                    "primaryDocument": ["exm-20230401.htm"],
                },
            )
        raise AssertionError(f"unexpected request {url}")

    client = make_client(handler)
    refs = client.list_filings(
        1, forms=["10-K", "10-Q"], start=date(2023, 1, 1), end=date(2024, 12, 31), ticker="EXM"
    )
    assert [r.accession for r in refs] == [
        "0000000001-24-000010",
        "0000000001-23-000005",
        "0000000001-23-000002",
    ]
    assert not any("submissions-001" in u for u in requested)  # page outside the window skipped
    k = refs[0]
    assert k.company == "EXAMPLE CORP" and k.fiscal_year_end == "0930" and k.ticker == "EXM"
    assert k.url == "https://www.sec.gov/Archives/edgar/data/1/000000000124000010/exm-20240928.htm"
    with_amend = client.list_filings(
        1, forms=["10-Q"], start=date(2024, 1, 1), end=date(2024, 12, 31), include_amendments=True
    )
    assert [r.form_type for r in with_amend] == ["10-Q/A"]


def test_download_is_cached_and_idempotent(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"<html><body>Item 1. Business</body></html>")

    client = make_client(handler)
    ref = FilingRef(
        cik=1,
        ticker="EXM",
        company="EXAMPLE CORP",
        form_type="10-K",
        accession="0000000001-24-000010",
        filing_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        primary_document="exm.htm",
        fiscal_year_end="0930",
    )
    p1 = client.download(ref, tmp_path)
    p2 = client.download(ref, tmp_path)
    assert p1 == p2 and p1.read_bytes().startswith(b"<html>")
    assert len(calls) == 1
    meta = json.loads((p1.parent / "filing.json").read_text())
    assert meta["source_url"] == ref.url
    ((cached_ref, path),) = load_cached_refs(tmp_path)
    assert cached_ref == ref and path == p1


def test_ticker_resolution(tmp_path):
    tmap = {
        "BRK-B": (1067983, "BERKSHIRE HATHAWAY INC"),
        "GOOGL": (1652044, "Alphabet Inc."),
        "GOOG": (1652044, "Alphabet Inc."),
    }
    companies, unknown = resolve_companies(["brk.b", "GOOGL", "GOOG", "NOPE"], tmap)
    assert [(c.ticker, c.cik) for c in companies] == [("BRK-B", 1067983), ("GOOGL", 1652044)]
    assert unknown == ["NOPE"]
    assert normalize_ticker(" brk.b ") == "BRK-B"


def test_default_company_list_is_big_enough():
    tickers = load_tickers("config/companies.yaml")
    assert len(tickers) >= 100 and len(set(tickers)) == len(tickers)
