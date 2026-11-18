"""Detect the Items (sections) of a 10-K or 10-Q.

Headings like "Item 7. Management's Discussion..." appear twice in most filings: once
in the table of contents and once in the body, plus stray cross-references. We pick
headings with a small dynamic program:

  * candidates = short text blocks that start with "Item N[A-C]" (TOC links and
    page-numbered rows are excluded up front);
  * choose a subsequence whose positions increase AND whose items follow the
    canonical order (1, 1A, 1B, ... 16 for a 10-K; Part I 1-4 then Part II 1-6 for a 10-Q);
  * score = sum over resulting sections of log(1 + length/200), plus a small bonus
    when the heading text matches the canonical item title.

The log is concave, so a TOC cluster (dozens of 50-character "sections") scores far
below the real body headings, and a 10-Q's repeated "Item 1" is resolved to Part I or
Part II by whichever assignment keeps the order valid. If fewer than three Items are
found (some banks file a narrative annual report with a cross-reference index), we
fall back to headings that match canonical titles ("Risk Factors", ...).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from filingqa.ingest.html_parser import Block

TENK_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer "
    "Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TENK_ORDER = list(TENK_TITLES)

TENQ_TITLES: dict[tuple[str, str], str] = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "3"): "Defaults Upon Senior Securities",
    ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information",
    ("II", "6"): "Exhibits",
}
TENQ_ORDER = list(TENQ_TITLES)

ITEM_RE = re.compile(
    r"^\s*(?:part\s+(?P<part>iv|i{1,3})\b[\s,.:;\-–—]*)?"
    r"items?\s*(?P<num>1[0-6]|[1-9])(?P<letter>[a-c])?(?![0-9a-z])"
    r"\s*[.:\-–—]?\s*(?P<title>.*)$",
    re.I | re.S,
)
PART_RE = re.compile(r"^\s*part\s+(?P<part>iv|i{1,3})\b\s*[.:\-–—]?\s*(?P<rest>.*)$", re.I | re.S)
_STOP = {"and", "of", "the", "for", "in", "on", "with", "to", "about", "that", "a"}

MAX_HEADING_CHARS = 200


@dataclass
class Section:
    section_id: str  # "cover", "1A", "7", "I-2", "II-1A", "document"
    title: str
    part: str | None
    char_start: int
    char_end: int


@dataclass
class _Cand:
    pos: int
    order: int
    section_id: str
    part: str | None
    title: str
    bonus: float


def _norm(s: str) -> str:
    return s.replace("’", "'").replace("‘", "'").lower()


def _title_matches(text: str, canonical: str) -> bool:
    words = [w for w in re.findall(r"[a-z]+", _norm(canonical)) if w not in _STOP][:2]
    t = _norm(text)
    return bool(words) and all(w in t for w in words)


def _is_10q(form_type: str) -> bool:
    return form_type.upper().startswith("10-Q")


def _item_candidates(blocks: Sequence[Block], form_type: str) -> list[_Cand]:
    tenq = _is_10q(form_type)
    part_marks: list[tuple[int, str]] = []
    for b in blocks:
        if b.kind == "text" and not b.toc_like and len(b.text) <= 150:
            pm = PART_RE.match(b.text)
            if pm and not ITEM_RE.match(b.text):
                part_marks.append((b.char_start, pm.group("part").upper()))

    def part_hint(pos: int) -> str | None:
        hint = None
        for p, part in part_marks:
            if p <= pos:
                hint = part
            else:
                break
        return hint

    cands: list[_Cand] = []
    for i, b in enumerate(blocks):
        if b.kind != "text" or b.toc_like:
            continue
        m = ITEM_RE.match(b.text)
        if not m:
            continue
        key = m.group("num") + (m.group("letter") or "").upper()
        title_text = (m.group("title") or "").strip()
        if not title_text and i + 1 < len(blocks):
            nxt = blocks[i + 1]
            if nxt.kind == "text" and len(nxt.text) <= MAX_HEADING_CHARS:
                title_text = nxt.text
        explicit_part = m.group("part").upper() if m.group("part") else None
        if tenq:
            parts = [explicit_part] if explicit_part else ["I", "II"]
            for part in parts:
                if (part, key) not in TENQ_TITLES:
                    continue
                canonical = TENQ_TITLES[(part, key)]
                tm = _title_matches(title_text, canonical)
                if len(b.text) > MAX_HEADING_CHARS and not tm:
                    continue
                bonus = 0.5 if tm else 0.0
                hint = part_hint(b.char_start)
                if not explicit_part and hint and hint != part:
                    bonus -= 1.0
                cands.append(
                    _Cand(
                        pos=b.char_start,
                        order=TENQ_ORDER.index((part, key)),
                        section_id=f"{part}-{key}",
                        part=part,
                        title=f"Part {part}, Item {key}. {canonical}",
                        bonus=bonus,
                    )
                )
        else:
            if key not in TENK_TITLES:
                continue
            canonical = TENK_TITLES[key]
            tm = _title_matches(title_text, canonical)
            if len(b.text) > MAX_HEADING_CHARS and not tm:
                continue
            cands.append(
                _Cand(
                    pos=b.char_start,
                    order=TENK_ORDER.index(key),
                    section_id=key,
                    part=None,
                    title=f"Item {key}. {canonical}",
                    bonus=0.5 if tm else 0.0,
                )
            )
    return cands


_KEYWORDS_10K = [
    ("business", "1"),
    ("risk factors", "1A"),
    ("unresolved staff comments", "1B"),
    ("cybersecurity", "1C"),
    ("properties", "2"),
    ("legal proceedings", "3"),
    ("mine safety disclosures", "4"),
    ("market for registrant", "5"),
    ("management's discussion and analysis", "7"),
    ("quantitative and qualitative disclosures about market risk", "7A"),
    ("financial statements and supplementary data", "8"),
    ("changes in and disagreements with accountants", "9"),
    ("controls and procedures", "9A"),
    ("other information", "9B"),
    ("directors, executive officers and corporate governance", "10"),
    ("executive compensation", "11"),
    ("security ownership of certain beneficial owners", "12"),
    ("certain relationships and related transactions", "13"),
    ("principal accountant fees and services", "14"),
    ("exhibits", "15"),
]
_KEYWORDS_10Q = [
    ("financial statements", ("I", "1")),
    ("management's discussion and analysis", ("I", "2")),
    ("quantitative and qualitative disclosures about market risk", ("I", "3")),
    ("controls and procedures", ("I", "4")),
    ("legal proceedings", ("II", "1")),
    ("risk factors", ("II", "1A")),
    ("unregistered sales of equity securities", ("II", "2")),
    ("defaults upon senior securities", ("II", "3")),
    ("mine safety disclosures", ("II", "4")),
    ("other information", ("II", "5")),
    ("exhibits", ("II", "6")),
]


def _keyword_candidates(blocks: Sequence[Block], form_type: str) -> list[_Cand]:
    cands: list[_Cand] = []
    tenq = _is_10q(form_type)
    for b in blocks:
        t = b.text.strip()
        if b.kind != "text" or b.toc_like or len(t) > 100 or t.endswith(".") or not t[:1].isupper():
            continue
        nt = _norm(t)
        if tenq:
            for kw, (part, key) in _KEYWORDS_10Q:
                if nt.startswith(kw):
                    cands.append(
                        _Cand(
                            pos=b.char_start,
                            order=TENQ_ORDER.index((part, key)),
                            section_id=f"{part}-{key}",
                            part=part,
                            title=f"Part {part}, Item {key}. {TENQ_TITLES[(part, key)]}",
                            bonus=0.0,
                        )
                    )
                    break
        else:
            for kw, key in _KEYWORDS_10K:
                if nt.startswith(kw) and (kw != "business" or len(nt) <= 20):
                    cands.append(
                        _Cand(
                            pos=b.char_start,
                            order=TENK_ORDER.index(key),
                            section_id=key,
                            part=None,
                            title=f"Item {key}. {TENK_TITLES[key]}",
                            bonus=0.0,
                        )
                    )
                    break
    return cands


def _score(length: int) -> float:
    return math.log1p(max(length, 0) / 200.0)


def select_headings(cands: list[_Cand], doc_len: int) -> list[_Cand]:
    """Best ordered subsequence of candidates (see module docstring)."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: (c.pos, c.order))[:600]
    n = len(cands)
    dp = [0.0] * n
    prev = [-1] * n
    for j, c in enumerate(cands):
        best, arg = _score(c.pos), -1  # c is the first heading; [0, pos) is the cover
        for i in range(j):
            ci = cands[i]
            if ci.pos < c.pos and ci.order < c.order:
                v = dp[i] + _score(c.pos - ci.pos)
                if v > best:
                    best, arg = v, i
        dp[j] = best + c.bonus
        prev[j] = arg
    end = max(range(n), key=lambda j: dp[j] + _score(doc_len - cands[j].pos))
    chain: list[_Cand] = []
    while end != -1:
        chain.append(cands[end])
        end = prev[end]
    return chain[::-1]


def detect_sections(blocks: Sequence[Block], form_type: str, doc_len: int) -> list[Section]:
    chain = select_headings(_item_candidates(blocks, form_type), doc_len)
    if len(chain) < 3:
        alt = select_headings(_keyword_candidates(blocks, form_type), doc_len)
        if len(alt) > len(chain):
            chain = alt
    if not chain:
        return [Section("document", "Full document (no Items detected)", None, 0, doc_len)]
    sections: list[Section] = []
    if chain[0].pos > 0:
        sections.append(Section("cover", "Cover page and front matter", None, 0, chain[0].pos))
    for k, c in enumerate(chain):
        end = chain[k + 1].pos if k + 1 < len(chain) else doc_len
        sections.append(Section(c.section_id, c.title, c.part, c.pos, end))
    return sections
