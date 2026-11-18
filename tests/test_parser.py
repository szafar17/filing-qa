from __future__ import annotations

from datetime import date

from filingqa.ingest.fiscal import infer_fiscal_period, period_label
from filingqa.ingest.html_parser import Block, parse_filing
from filingqa.ingest.sections import detect_sections

TENK_ITEMS = [
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "9C",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
]


def ids(parsed):
    return [s.section_id for s in parsed.sections]


def test_10k_items_detected_in_order_and_toc_skipped(acme):
    assert ids(acme) == ["cover", *TENK_ITEMS]
    # The TOC (hyperlinked rows with page numbers) stays in the cover, not in Item 1.
    item1 = next(s for s in acme.sections if s.section_id == "1")
    assert acme.text[item1.char_start :].startswith("Item 1. Business")
    cover = acme.sections[0]
    assert "TABLE OF CONTENTS" in acme.text[cover.char_start : cover.char_end]


def test_sections_tile_the_document(acme, bluewater, cobalt):
    for parsed in (acme, bluewater, cobalt):
        secs = parsed.sections
        assert secs[0].char_start == 0
        assert secs[-1].char_end == len(parsed.text)
        for a, b in zip(secs, secs[1:], strict=False):
            assert a.char_end == b.char_start


def test_ixbrl_hidden_header_removed_and_dei_extracted(bluewater):
    assert bluewater.dei["DocumentFiscalYearFocus"] == "2024"
    assert bluewater.dei["DocumentFiscalPeriodFocus"] == "FY"
    assert bluewater.dei["TradingSymbol"] == "ZZBWF"
    # contents of the display:none ix:header must not leak into the text
    assert "0009900002" not in bluewater.text
    assert "2023-07-01" not in bluewater.text
    assert "Item 7. Management's Discussion" in next(
        s.title for s in bluewater.sections if s.section_id == "7"
    )


def test_uppercase_headings_with_curly_apostrophes(bluewater):
    assert ids(bluewater) == [
        "cover",
        "1",
        "1A",
        "1B",
        "1C",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "7A",
        "8",
        "9",
        "9A",
        "9B",
        "10",
        "15",
    ]


def test_10q_part_items_resolved(cobalt):
    assert ids(cobalt) == [
        "cover",
        "I-1",
        "I-2",
        "I-3",
        "I-4",
        "II-1",
        "II-1A",
        "II-2",
        "II-3",
        "II-4",
        "II-5",
        "II-6",
    ]
    titles = {s.section_id: s.title for s in cobalt.sections}
    assert titles["I-1"] == "Part I, Item 1. Financial Statements"
    assert titles["II-1"] == "Part II, Item 1. Legal Proceedings"


def test_long_paragraph_that_starts_with_item_heading(cobalt):
    sec = next(s for s in cobalt.sections if s.section_id == "II-5")
    body = cobalt.text[sec.char_start : sec.char_end]
    assert body.startswith("Item 5. Other Information. During the three months")
    assert "majority voting" in body


def test_page_footers_and_numbers_removed(acme, cobalt):
    assert "2024 Form 10-K |" not in acme.text
    assert "Q2 2024 Form 10-Q |" not in cobalt.text


def test_blocks_have_exact_offsets(acme):
    for b in acme.blocks:
        assert acme.text[b.char_start : b.char_end] == b.text


def test_table_titles_attached(acme):
    tables = [b for b in acme.blocks if b.kind == "table"]
    titles = [t.table_title for t in tables]
    assert "(In millions, except per share amounts)" in titles


def test_keyword_fallback_when_no_item_headings():
    html = b"""<html><body>
    <p>Example Bank Corp annual report</p>
    <p>Risk Factors</p><p>%s</p>
    <p>Management's discussion and analysis</p><p>%s</p>
    <p>Controls and Procedures</p><p>%s</p>
    </body></html>""" % (
        b"Credit risk is our largest risk. " * 30,
        b"Net interest income rose. " * 30,
        b"Controls were effective. " * 20,
    )
    parsed = parse_filing(html, "10-K")
    assert ids(parsed)[1:] == ["1A", "7", "9A"]


def test_no_items_gives_single_document_section():
    parsed = parse_filing(b"<html><body><p>Just some text.</p></body></html>", "10-K")
    assert ids(parsed) == ["document"]


def test_plain_text_filing():
    raw = b"ITEM 1. BUSINESS\n\nWe make widgets.\n\nITEM 1A. RISK FACTORS\n\nWidgets may break.\n"
    parsed = parse_filing(raw, "10-K")
    assert ids(parsed) == ["1", "1A"]


def test_toc_like_blocks_never_become_headings():
    blocks = []
    pos = 0
    for text, toc in [
        ("Item 1. Business", True),
        ("Item 7. MD&A", True),
        ("Item 1. Business", False),
        ("x" * 3000, False),
        ("Item 7. Management's", False),
        ("y" * 3000, False),
    ]:
        blocks.append(Block(kind="text", text=text, toc_like=toc, char_start=pos, char_end=pos + len(text)))
        pos += len(text) + 2
    secs = detect_sections(blocks, "10-K", pos)
    assert [s.section_id for s in secs] == ["cover", "1", "7"]
    assert secs[1].char_start == blocks[2].char_start


# --- fiscal period labels -------------------------------------------------------------


def test_fiscal_from_dei_wins():
    dei = {"DocumentFiscalYearFocus": "2023", "DocumentFiscalPeriodFocus": "FY"}
    assert infer_fiscal_period("10-K", date(2024, 2, 3), "0203", dei) == (2023, "FY")


def test_fiscal_fallback_september_year_end():
    # Apple-style: FY ends late September; Q1 ends in late December of the prior calendar year
    assert infer_fiscal_period("10-K", date(2023, 9, 30), "0930") == (2023, "FY")
    assert infer_fiscal_period("10-Q", date(2023, 12, 30), "0930") == (2024, "Q1")
    # 52/53-week calendar: a quarter that ends on July 1 belongs to the June quarter
    assert infer_fiscal_period("10-Q", date(2023, 7, 1), "0930") == (2023, "Q3")


def test_fiscal_fallback_calendar_and_january_year_ends():
    assert infer_fiscal_period("10-Q", date(2024, 6, 30), "1231") == (2024, "Q2")
    assert infer_fiscal_period("10-Q", date(2024, 4, 30), "0131") == (2025, "Q1")
    assert infer_fiscal_period("10-K", None, "1231") == (None, None)


def test_period_label():
    assert period_label(2024, "FY") == "FY2024"
    assert period_label(2024, "Q2") == "Q2 FY2024"
    assert period_label(None, "Q2") is None
