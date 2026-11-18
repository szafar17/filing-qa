"""Fiscal year / period labels for a filing.

Preferred source: inline-XBRL DEI facts (DocumentFiscalYearFocus / DocumentFiscalPeriodFocus),
which are the company's own labels. Fallback: derive from the period-of-report date and
the company's fiscal-year-end (MMDD) from the EDGAR submissions API. 52/53-week fiscal
calendars end a few days into the next month (Apple's Q3 FY2023 ended July 1, 2023), so
period dates in the first week of a month are treated as the previous month's end.
Retailers that name fiscal years by their start year (e.g. a year ending Feb 2024 called
"fiscal 2023") can be off by one in the fallback; the DEI facts get those right.
"""

from __future__ import annotations

from datetime import date, timedelta

_VALID_PERIODS = {"FY", "Q1", "Q2", "Q3", "Q4"}


def _effective_month(d: date) -> tuple[int, int]:
    if d.day <= 7:
        d = d.replace(day=1) - timedelta(days=1)
    return d.year, d.month


def infer_fiscal_period(
    form_type: str,
    period_of_report: date | None,
    fiscal_year_end: str | None = None,
    dei: dict[str, str] | None = None,
) -> tuple[int | None, str | None]:
    dei = dei or {}
    fy_raw = (dei.get("DocumentFiscalYearFocus") or "").strip()
    fp_raw = (dei.get("DocumentFiscalPeriodFocus") or "").strip().upper()
    if fy_raw.isdigit() and len(fy_raw) == 4 and fp_raw in _VALID_PERIODS:
        return int(fy_raw), fp_raw

    if period_of_report is None:
        return None, None
    year, month = _effective_month(period_of_report)
    fye_month = 12
    if fiscal_year_end and len(fiscal_year_end) == 4 and fiscal_year_end.isdigit():
        mm, dd = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
        if 1 <= mm <= 12:
            fye_month = mm
            if dd <= 7:  # fiscal year ends in the first days of a month: use the prior month
                fye_month = 12 if mm == 1 else mm - 1
    fiscal_year = year + 1 if month > fye_month else year
    if form_type.upper().startswith("10-K"):
        return fiscal_year, "FY"
    quarter = ((month - fye_month - 1) % 12) // 3 + 1
    return fiscal_year, f"Q{quarter}"


def period_label(fiscal_year: int | None, fiscal_period: str | None) -> str | None:
    if fiscal_year is None:
        return None
    if fiscal_period in (None, "FY"):
        return f"FY{fiscal_year}"
    return f"{fiscal_period} FY{fiscal_year}"
