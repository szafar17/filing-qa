"""Turn a 10-K/10-Q primary document (HTML or inline XBRL) into clean text blocks.

Output is a ParsedFiling: the full normalized text, the ordered blocks it is made of
(paragraphs and flattened tables, each with character offsets into the full text),
detected Items/sections, and DEI facts from the inline-XBRL header (fiscal year,
fiscal period, ticker...). Character offsets are what make citations checkable:
chunk text is always an exact slice of ParsedFiling.text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from lxml import etree
from lxml import html as lxml_html

from filingqa.ingest.sections import Section, detect_sections
from filingqa.ingest.tables import flatten_table
from filingqa.textutil import normalize_ws

PARSER_VERSION = "1"

BLOCK_TAGS = frozenset(
    """p div br li ul ol h1 h2 h3 h4 h5 h6 tr section article header footer blockquote pre
    center dl dt dd hr body html form fieldset address figure figcaption main nav aside caption
    page""".split()
)
SKIP_TAGS = frozenset(
    """script style head title noscript meta link ix:header ix:hidden ix:references
    ix:resources xbrli:context xbrli:unit""".split()
)
DEI_FIELDS = (
    "dei:DocumentType",
    "dei:DocumentPeriodEndDate",
    "dei:DocumentFiscalYearFocus",
    "dei:DocumentFiscalPeriodFocus",
    "dei:CurrentFiscalYearEndDate",
    "dei:EntityRegistrantName",
    "dei:EntityCentralIndexKey",
    "dei:TradingSymbol",
)
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_PAGE_ONLY = re.compile(r"^(?:page\s+)?(?:\d{1,3}|[ivxlc]{1,6})$", re.I)


@dataclass
class Block:
    kind: str  # "text" | "table"
    text: str
    toc_like: bool = False  # came from a table-of-contents link or page-numbered row
    table_header: str | None = None  # "Columns: ..." line for data tables
    table_title: str | None = None  # nearest short paragraph before the table
    char_start: int = 0
    char_end: int = 0


@dataclass
class ParsedFiling:
    text: str
    blocks: list[Block]
    sections: list[Section]
    dei: dict[str, str] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION


class _Linearizer:
    """Walk the DOM in document order, emitting a block at every block-level boundary."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._link_chars = 0
        self._chars = 0

    def _add(self, text: str, in_link: bool) -> None:
        if not text:
            return
        self._buf.append(text)
        n = len(text.strip())
        self._chars += n
        if in_link:
            self._link_chars += n

    def _flush(self) -> None:
        if self._buf:
            text = normalize_ws("".join(self._buf))
            if text:
                toc = self._chars > 0 and self._link_chars >= 0.5 * self._chars
                self.blocks.append(Block(kind="text", text=text, toc_like=toc))
        self._buf = []
        self._link_chars = 0
        self._chars = 0

    def _table(self, el: lxml_html.HtmlElement) -> None:
        flat = flatten_table(el)
        if flat.is_data:
            if flat.lines:
                self.blocks.append(
                    Block(kind="table", text="\n".join(flat.lines), table_header=flat.header_line)
                )
        else:
            for text, toc in flat.layout_rows:
                t = normalize_ws(text)
                if t:
                    self.blocks.append(Block(kind="text", text=t, toc_like=toc))

    def run(self, root: lxml_html.HtmlElement) -> list[Block]:
        # Explicit stack instead of recursion: some filings nest thousands of <font> tags.
        stack: list[tuple] = [("el", root, False)]
        while stack:
            item = stack.pop()
            kind = item[0]
            if kind == "text":
                self._add(item[1], item[2])
                continue
            if kind == "flush":
                self._flush()
                continue
            el, in_link = item[1], item[2]
            tag = el.tag.lower() if isinstance(el.tag, str) else ""
            if not tag or tag in SKIP_TAGS or _HIDDEN.search(el.get("style") or ""):
                continue
            if tag == "table":
                self._flush()
                self._table(el)
                continue
            link = in_link or (tag == "a" and (el.get("href") or "").startswith("#"))
            is_block = tag in BLOCK_TAGS
            # Push in reverse execution order: flush, text, (child, tail)..., flush.
            if is_block:
                stack.append(("flush",))
            for child in reversed(list(el)):
                if child.tail:
                    stack.append(("text", child.tail, link))
                stack.append(("el", child, link))
            if el.text:
                stack.append(("text", el.text, link))
            if is_block:
                stack.append(("flush",))
        self._flush()
        return self.blocks


def extract_dei(root: lxml_html.HtmlElement) -> dict[str, str]:
    """Pull document/entity facts from inline XBRL (often inside the hidden ix:header)."""
    found: dict[str, str] = {}
    for el in root.iter():
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        if not tag.endswith("nonnumeric"):
            continue
        name = el.get("name") or ""
        if name in DEI_FIELDS and name not in found:
            value = normalize_ws(el.text_content())
            if value:
                found[name.split(":", 1)[1]] = value
    return found


def _drop_repeated_furniture(blocks: list[Block]) -> list[Block]:
    """Remove page numbers and running headers/footers ("Acme Corp | 2024 Form 10-K | 17")."""

    def key(b: Block) -> str:
        return re.sub(r"\d+", "#", b.text.lower())

    counts = Counter(key(b) for b in blocks if b.kind == "text" and len(b.text) < 100)
    out: list[Block] = []
    for b in blocks:
        if b.kind == "text":
            t = b.text.strip()
            if _PAGE_ONLY.match(t):
                continue
            k = key(b)
            if (
                len(t) < 100
                and counts[k] >= 4
                and (any(ch.isdigit() for ch in t) or k == "table of contents")
            ):
                continue
            if t.lower() in ("table of contents", "index") and counts[k] >= 2:
                continue
        out.append(b)
    return out


def _looks_like_html(raw: bytes) -> bool:
    head = raw[:4096].lower()
    return b"<html" in head or b"<body" in head or b"<div" in head or b"<?xml" in head


def parse_filing(raw: bytes, form_type: str) -> ParsedFiling:
    """Parse a primary document (bytes) into blocks, sections and DEI facts."""
    dei: dict[str, str] = {}
    if _looks_like_html(raw):
        parser = lxml_html.HTMLParser(remove_comments=True, remove_pis=True, huge_tree=True)
        try:
            root = lxml_html.document_fromstring(raw, parser=parser)
        except (etree.ParserError, ValueError):
            root = lxml_html.document_fromstring(raw.decode("utf-8", "replace"), parser=parser)
        dei = extract_dei(root)
        blocks = _Linearizer().run(root)
    else:  # plain-text filing: paragraphs separated by blank lines
        text = raw.decode("utf-8", "replace")
        blocks = [Block(kind="text", text=normalize_ws(p)) for p in re.split(r"\n\s*\n", text)]
        blocks = [b for b in blocks if b.text]

    blocks = _drop_repeated_furniture(blocks)

    # Attach a title to each data table: the closest short paragraph just before it.
    for i, b in enumerate(blocks):
        if b.kind == "table" and i > 0 and blocks[i - 1].kind == "text":
            prev = blocks[i - 1].text
            if len(prev) <= 200:
                b.table_title = prev

    # Assemble the full text; blocks are separated by a blank line.
    parts: list[str] = []
    pos = 0
    for b in blocks:
        if parts:
            parts.append("\n\n")
            pos += 2
        b.char_start = pos
        parts.append(b.text)
        pos += len(b.text)
        b.char_end = pos
    full_text = "".join(parts)
    sections = detect_sections(blocks, form_type, len(full_text))
    return ParsedFiling(text=full_text, blocks=blocks, sections=sections, dei=dei)
