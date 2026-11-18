"""Flatten HTML tables from SEC filings into readable row text.

SEC financial tables are hostile to naive text extraction: the "$" sign sits in its
own cell, negative numbers are split into "(1,234" and ")", headers span several
physical columns via colspan, and empty spacer columns are everywhere. This module
rebuilds a logical table and emits one line per row:

    Columns: Year Ended Sept 30, 2023 | Year Ended Sept 30, 2022
    Net sales | Year Ended Sept 30, 2023: $383,285 | Year Ended Sept 30, 2022: $394,328

so that each row keeps its label, its values, and the column each value belongs to.
Tables with no real numeric rows are layout tables (SEC filings use tables to lay
out headings and two-column text); those are returned as plain text lines instead.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from lxml import html as lxml_html

from filingqa.textutil import normalize_ws

_CURRENCY = {"$", "€", "£", "¥", "US$"}
_CLOSERS = re.compile(r"^\)?\s*%?\s*\)?$")
_NUM = re.compile(r"^[\(\-–—]?\s*(?:us)?[$€£¥]?\s*\(?\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*%?\s*\)?\*?$")
_NIL = {"—", "–", "-", "—%", "— %", "nm", "n/m", "n/a", "na", "*", "——", "--"}
_YEAR = re.compile(r"^(?:19|20)\d\d$")
_PAGE_NUM = re.compile(r"^(?:\d{1,3}|[ivxlc]{1,6}|page\s+\d{1,3})$", re.I)


@dataclass
class Cell:
    text: str
    col: int
    span: int = 1


@dataclass
class FlatTable:
    is_data: bool
    lines: list[str] = field(default_factory=list)  # data table: header line (optional) + rows
    header_line: str | None = None
    layout_rows: list[tuple[str, bool]] = field(default_factory=list)  # (text, toc_like)
    n_rows: int = 0


def is_value(text: str) -> bool:
    t = text.strip().lower()
    return bool(_NUM.match(t)) or t in _NIL


def is_year(text: str) -> bool:
    return bool(_YEAR.match(text.strip()))


def is_numeric_value(text: str) -> bool:
    """A value cell that carries data (not a bare year used as a column header)."""
    return is_value(text) and not is_year(text)


def _row_elements(table: lxml_html.HtmlElement) -> list[lxml_html.HtmlElement]:
    return table.xpath("./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr")


def extract_rows(table: lxml_html.HtmlElement) -> list[tuple[list[Cell], bool]]:
    """Return rows as lists of non-empty cells (with grid columns) plus a toc_like flag."""
    rows: list[tuple[list[Cell], bool]] = []
    for tr in _row_elements(table):
        col = 0
        cells: list[Cell] = []
        has_anchor_link = bool(tr.xpath(".//a[starts-with(@href, '#')]"))
        for td in tr.xpath("./td | ./th"):
            try:
                span = int(str(td.get("colspan") or "1").strip() or 1)
            except ValueError:
                span = 1
            span = max(1, min(span, 50))
            text = normalize_ws(td.text_content())
            cells.append(Cell(text=text, col=col, span=span))
            col += span
        merged = merge_fragments(cells)
        if merged:
            last = merged[-1].text
            toc_like = has_anchor_link or (len(merged) >= 2 and bool(_PAGE_NUM.match(last)))
            rows.append((merged, toc_like))
    return rows


def merge_fragments(cells: list[Cell]) -> list[Cell]:
    """Glue "$" onto the following value and ")" / "%" onto the preceding one; drop empties."""
    out: list[Cell] = []
    prefix = ""
    prefix_col: int | None = None
    for c in cells:
        t = c.text
        if not t:
            continue
        if t in _CURRENCY:
            prefix = t
            prefix_col = c.col
            continue
        if out and _CLOSERS.match(t) and t:
            out[-1] = Cell(text=out[-1].text + t, col=out[-1].col, span=out[-1].span)
            continue
        if prefix and prefix_col is not None:
            # The merged "$value" spans from the currency cell through the value cell, because
            # SEC tables often put the column header over the "$" cell, not over the number.
            out.append(Cell(text=prefix + t, col=prefix_col, span=c.col + c.span - prefix_col))
            prefix = ""
            prefix_col = None
            continue
        out.append(Cell(text=t, col=c.col, span=c.span))
    if prefix and prefix_col is not None:  # dangling currency symbol; keep it rather than lose it
        out.append(Cell(text=prefix, col=prefix_col))
    return out


def _covers(cell: Cell, col: int, span: int) -> bool:
    return cell.col < col + span and col < cell.col + cell.span


def _distance(cell: Cell, col: int, span: int) -> int:
    if _covers(cell, col, span):
        return 0
    if cell.col + cell.span <= col:
        return col - (cell.col + cell.span - 1)
    return cell.col - (col + span - 1)


def value_clusters(rows: list[list[Cell]]) -> list[tuple[int, int]]:
    """Group the physical columns used by values into logical columns.

    Each value cell covers [col, col+span). Spans that overlap across rows belong to the
    same logical column, e.g. "$812.9" (cols 1-2) and "471.7" (col 2) are one column.
    Returns (start, end_exclusive) per logical column, left to right.
    """
    spans = sorted((c.col, c.col + c.span) for row in rows for c in row)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def header_for(value: Cell, header_rows: list[list[Cell]], clusters: list[tuple[int, int]]) -> str:
    """Join, top to bottom, the header cell of each header row that sits over ``value``.

    The value is widened to its logical column (see value_clusters). A header cell
    matches if it covers that column. A single-column header (typically a year) that
    is misaligned by one column is accepted as a fallback; ties go left, since SEC
    tables usually put the header over the "$" column that precedes the number.
    """
    col, span = value.col, value.span
    for a, b in clusters:
        if a < value.col + value.span and value.col < b:
            col, span = a, b - a
            break
    parts: list[str] = []
    for row in header_rows:
        covering = [c for c in row if _covers(c, col, span)]
        best: Cell | None = None
        if covering:
            best = max(covering, key=lambda c: min(c.col + c.span, col + span) - max(c.col, col))
        else:
            near = [c for c in row if c.span == 1 and _distance(c, col, span) == 1]
            if near:
                best = min(near, key=lambda c: c.col)
        if best is not None and (not parts or parts[-1] != best.text):
            parts.append(best.text)
    return " ".join(parts)


def flatten_table(table: lxml_html.HtmlElement) -> FlatTable:
    rows = extract_rows(table)
    numeric_rows = [r for r, _ in rows if any(is_numeric_value(c.text) for c in r)]
    if len(numeric_rows) < 2:
        layout = [(" ".join(c.text for c in r), toc) for r, toc in rows]
        return FlatTable(is_data=False, layout_rows=[(t, toc) for t, toc in layout if t], n_rows=len(rows))

    # Label column = most common column of the first (non-value) cell in rows with numbers.
    label_cols = Counter(r[0].col for r in numeric_rows if not is_value(r[0].text))
    label_col = label_cols.most_common(1)[0][0] if label_cols else 0

    # Pass 1: classify rows and split them into header blocks (a header row that follows
    # data rows starts a new sub-table with its own column headers).
    kinds: list[tuple[str, int, list[Cell], Cell, bool]] = []  # (kind, block, values, first, has_label)
    block = 0
    seen_data = False
    for cells, _toc in rows:
        first = cells[0]
        has_label = (not is_value(first.text)) and first.col <= label_col
        values = cells[1:] if has_label else cells
        if values and not any(is_numeric_value(c.text) for c in values):
            if seen_data:
                block += 1
                seen_data = False
            kinds.append(("header", block, values, first, has_label))
        elif not values:
            kinds.append(("label", block, values, first, has_label))
        else:
            seen_data = True
            kinds.append(("data", block, values, first, has_label))

    clusters_by_block: dict[int, list[tuple[int, int]]] = {}
    for blk in {k[1] for k in kinds}:
        clusters_by_block[blk] = value_clusters([k[2] for k in kinds if k[1] == blk and k[0] == "data"])

    # Pass 2: emit lines.
    header_rows: list[list[Cell]] = []
    current_block = -1
    lines: list[str] = []
    first_header_line: str | None = None
    columns_emitted = False
    for kind, blk, values, first, has_label in kinds:
        if blk != current_block:
            current_block, header_rows, columns_emitted = blk, [], False
        clusters = clusters_by_block[blk]
        if kind == "header":
            if has_label:
                lines.append(first.text)  # units caption such as "(In millions)"
            header_rows.append(values)
            continue
        if kind == "label":
            lines.append(first.text)  # section label row, e.g. "Operating expenses:"
            continue
        if not columns_emitted:
            names: list[str] = []
            for c in values:
                h = header_for(c, header_rows, clusters)
                if h and h not in names:
                    names.append(h)
            if names:
                hl = "Columns: " + " | ".join(names)
                lines.append(hl)
                if first_header_line is None:
                    first_header_line = hl
            columns_emitted = True
        parts: list[str] = []
        for c in values:
            h = header_for(c, header_rows, clusters)
            parts.append(f"{h}: {c.text}" if h else c.text)
        label = first.text if has_label else ""
        lines.append(" | ".join([label, *parts]) if label else " | ".join(parts))

    return FlatTable(
        is_data=True,
        lines=lines,
        header_line=first_header_line,
        n_rows=len(rows),
    )
