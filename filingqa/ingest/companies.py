"""Company universe: tickers from config/companies.yaml resolved to SEC CIKs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str


def normalize_ticker(t: str) -> str:
    # SEC's ticker file writes share classes with a dash: BRK.B -> BRK-B
    return t.strip().upper().replace(".", "-")


def load_tickers(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    seen: list[str] = []
    for t in data.get("tickers", []):
        nt = normalize_ticker(str(t))
        if nt and nt not in seen:
            seen.append(nt)
    return seen


def resolve_companies(
    tickers: list[str], ticker_map: dict[str, tuple[int, str]]
) -> tuple[list[Company], list[str]]:
    """Map tickers to CIKs; returns (companies, unknown_tickers). Duplicate CIKs are dropped
    (e.g. GOOGL and GOOG are the same registrant)."""
    out: list[Company] = []
    unknown: list[str] = []
    seen_ciks: set[int] = set()
    for t in tickers:
        hit = ticker_map.get(normalize_ticker(t))
        if hit is None:
            unknown.append(t)
            continue
        cik, name = hit
        if cik in seen_ciks:
            continue
        seen_ciks.add(cik)
        out.append(Company(ticker=normalize_ticker(t), cik=cik, name=name))
    return out, unknown
