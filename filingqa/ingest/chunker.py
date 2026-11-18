"""Section-aware chunking with exact character offsets.

Rules:
  * A chunk never crosses a section (Item) boundary.
  * Units are sentences (for prose) and rows (for tables); chunks are packed up to
    ``target_chars`` and only exceed it when a single unit is larger (hard cap
    ``max_chars``, long sentences are split at whitespace).
  * Consecutive prose chunks overlap by up to ``overlap_chars`` of whole sentences.
    Table rows are not overlapped; instead a chunk that continues a table gets the
    table's title and "Columns:" line in its context header.
  * ``chunk.text == filing_text[char_start:char_end]`` always holds, so a quoted
    citation can be located in the original filing text.
  * The context header (company, ticker, form, fiscal period, filing date, section,
    table header) is embedded and keyword-indexed together with the text, but kept
    out of the text itself so offsets stay exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from filingqa.config import ChunkingConfig
from filingqa.ingest.fiscal import period_label
from filingqa.ingest.html_parser import ParsedFiling
from filingqa.ingest.sections import Section
from filingqa.textutil import sentence_spans

CHUNKER_VERSION = "1"


@dataclass
class FilingMeta:
    accession: str
    cik: int
    company: str
    ticker: str | None
    form_type: str
    filing_date: date | None
    period_of_report: date | None
    fiscal_year: int | None
    fiscal_period: str | None
    source_url: str
    synthetic: bool = False


@dataclass
class Chunk:
    chunk_uid: str
    accession: str
    chunk_index: int
    section_id: str
    section_title: str
    section_chunk_index: int
    char_start: int
    char_end: int
    text: str
    context_header: str
    has_table: bool


@dataclass
class _Unit:
    start: int
    end: int
    kind: str  # "text" | "row" | "table_header"
    block_idx: int


def filing_header(meta: FilingMeta) -> str:
    who = meta.company + (f" ({meta.ticker})" if meta.ticker else "")
    parts = [who, meta.form_type]
    label = period_label(meta.fiscal_year, meta.fiscal_period)
    if label:
        parts.append(f"fiscal {label}")
    if meta.period_of_report:
        parts.append(f"period ended {meta.period_of_report.isoformat()}")
    if meta.filing_date:
        parts.append(f"filed {meta.filing_date.isoformat()}")
    return " | ".join(parts)


def _split_long(start: int, end: int, text: str, max_chars: int) -> list[tuple[int, int]]:
    """Split [start, end) of ``text`` into pieces <= max_chars, breaking at whitespace."""
    out: list[tuple[int, int]] = []
    s = start
    while end - s > max_chars:
        cut = text.rfind(" ", s + max_chars // 2, s + max_chars)
        if cut <= s:
            cut = s + max_chars
        out.append((s, cut))
        s = cut
        while s < end and text[s] == " ":
            s += 1
    if end > s:
        out.append((s, end))
    return out


def _units_for_section(parsed: ParsedFiling, section: Section, cfg: ChunkingConfig) -> list[_Unit]:
    full = parsed.text
    units: list[_Unit] = []
    for bi, b in enumerate(parsed.blocks):
        if b.char_end <= section.char_start or b.char_start >= section.char_end:
            continue
        if b.kind == "table":
            off = b.char_start
            for line in b.text.split("\n"):
                if line:
                    kind = "table_header" if line.startswith("Columns: ") else "row"
                    for s, e in _split_long(off, off + len(line), full, cfg.max_chars):
                        units.append(_Unit(s, e, kind, bi))
                off += len(line) + 1
            continue
        for s, e in sentence_spans(b.text):
            for s2, e2 in _split_long(b.char_start + s, b.char_start + e, full, cfg.max_chars):
                units.append(_Unit(s2, e2, "text", bi))
    return units


def _pack(units: list[_Unit], cfg: ChunkingConfig) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    carried = 0  # how many leading units of `cur` are overlap from the previous chunk
    for u in units:
        if cur and len(cur) > carried and (u.end - cur[0].start) > cfg.target_chars:
            groups.append(cur)
            carry: list[_Unit] = []
            total = 0
            for v in reversed(cur):
                n = v.end - v.start
                if v.kind != "text" or total + n > cfg.overlap_chars:
                    break
                carry.insert(0, v)
                total += n
            if len(carry) >= len(cur):
                carry = []
            # never let overlap + next unit blow past the hard cap
            if carry and (u.end - carry[0].start) > cfg.max_chars:
                carry = []
            cur, carried = carry, len(carry)
        cur.append(u)
    if cur and len(cur) > carried:
        groups.append(cur)
    # merge a tiny tail into the previous chunk of the same section
    if len(groups) >= 2:
        last, prev = groups[-1], groups[-2]
        if (last[-1].end - last[0].start) < cfg.min_chars and (
            last[-1].end - prev[0].start
        ) <= cfg.max_chars + cfg.min_chars:
            merged = prev + [u for u in last if u.start >= prev[-1].end]
            groups[-2:] = [merged]
    return groups


def _table_context(parsed: ParsedFiling, group: list[_Unit], start: int, text: str) -> list[str]:
    lines: list[str] = []
    seen: set[int] = set()
    for u in group:
        if u.kind == "text" or u.block_idx in seen:
            continue
        seen.add(u.block_idx)
        b = parsed.blocks[u.block_idx]
        title_inside = u.block_idx > 0 and parsed.blocks[u.block_idx - 1].char_start >= start
        if b.table_title and not title_inside:
            prefix = "Table (continued)" if b.char_start < start else "Table"
            lines.append(f"{prefix}: {b.table_title}")
        if b.table_header and b.table_header not in text:
            lines.append(b.table_header)
    return lines


def chunk_filing(parsed: ParsedFiling, meta: FilingMeta, cfg: ChunkingConfig) -> list[Chunk]:
    header = filing_header(meta)
    chunks: list[Chunk] = []
    for section in parsed.sections:
        units = _units_for_section(parsed, section, cfg)
        for k, group in enumerate(_pack(units, cfg)):
            start, end = group[0].start, group[-1].end
            text = parsed.text[start:end]
            ctx = [header, f"Section: {section.title}", *_table_context(parsed, group, start, text)]
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_uid=f"{meta.accession}:{idx}",
                    accession=meta.accession,
                    chunk_index=idx,
                    section_id=section.section_id,
                    section_title=section.title,
                    section_chunk_index=k,
                    char_start=start,
                    char_end=end,
                    text=text,
                    context_header="\n".join(ctx),
                    has_table=any(u.kind != "text" for u in group),
                )
            )
    return chunks
