from __future__ import annotations

from filingqa.config import ChunkingConfig
from filingqa.ingest.chunker import chunk_filing, filing_header

from .conftest import filing_meta, parse_doc

SMALL = ChunkingConfig(target_chars=600, max_chars=900, overlap_chars=150, min_chars=120)


def chunks_for(key: str, cfg: ChunkingConfig = SMALL):
    parsed = parse_doc(key)
    return parsed, chunk_filing(parsed, filing_meta(key), cfg)


def test_chunk_text_is_exact_slice_of_filing_text():
    for key in ("acme", "bluewater", "cobalt"):
        parsed, chunks = chunks_for(key)
        assert chunks
        for c in chunks:
            assert c.text == parsed.text[c.char_start : c.char_end]


def test_chunks_never_cross_sections():
    parsed, chunks = chunks_for("acme")
    bounds = {s.section_id: (s.char_start, s.char_end) for s in parsed.sections}
    for c in chunks:
        s, e = bounds[c.section_id]
        assert s <= c.char_start < c.char_end <= e


def test_chunk_sizes_respect_hard_cap():
    _, chunks = chunks_for("acme")
    assert max(len(c.text) for c in chunks) <= SMALL.max_chars
    # sections longer than the target are split into several chunks
    assert sum(1 for c in chunks if c.section_id == "1") >= 3


def test_prose_chunks_overlap_by_whole_sentences():
    _, chunks = chunks_for("acme")
    item1 = [c for c in chunks if c.section_id == "1"]
    overlaps = [b.char_start < a.char_end for a, b in zip(item1, item1[1:], strict=False)]
    assert any(overlaps)
    for a, b in zip(item1, item1[1:], strict=False):
        assert b.char_end > a.char_end  # always moves forward
        assert a.char_end - b.char_start <= SMALL.overlap_chars


def test_metadata_and_uids():
    _, chunks = chunks_for("cobalt")
    c = chunks[0]
    assert c.chunk_uid == f"{c.accession}:0"
    assert [x.chunk_index for x in chunks] == list(range(len(chunks)))
    header = filing_header(filing_meta("cobalt"))
    assert header == (
        "Cobalt Energy Partners Corp. (SYNTHETIC) (ZZCOB) | 10-Q | fiscal Q2 FY2024 | "
        "period ended 2024-06-30 | filed 2024-08-07"
    )
    assert c.context_header.startswith(header + "\nSection: ")
    # per-section indices restart at 0
    for sid in {x.section_id for x in chunks}:
        idx = [x.section_chunk_index for x in chunks if x.section_id == sid]
        assert idx == list(range(len(idx)))


def test_table_continuation_chunks_get_title_and_columns_in_header():
    tiny = ChunkingConfig(target_chars=350, max_chars=600, overlap_chars=0, min_chars=50)
    parsed, chunks = chunks_for("acme", tiny)
    statement = [c for c in chunks if c.section_id == "8" and c.has_table]
    assert len(statement) >= 3
    continuation = [
        c
        for c in statement
        if c.text.startswith(
            (
                "Gross profit",
                "Operating",
                "Research",
                "Selling",
                "Interest",
                "Income",
                "Provision",
                "Net income",
                "Earnings",
                "Basic",
                "Diluted",
                "Weighted",
            )
        )
    ]
    assert continuation
    for c in continuation:
        assert "Table (continued): (In millions, except per share amounts)" in c.context_header
        assert "Columns: Year Ended December 31, 2024 |" in c.context_header


def test_rows_are_never_split():
    """Inside a table block, chunk boundaries fall only on row boundaries."""
    tiny = ChunkingConfig(target_chars=350, max_chars=600, overlap_chars=0, min_chars=50)
    for key in ("acme", "bluewater", "cobalt"):
        parsed, chunks = chunks_for(key, tiny)
        text = parsed.text
        for b in (b for b in parsed.blocks if b.kind == "table"):
            for c in chunks:
                if b.char_start < c.char_start < b.char_end:
                    assert text[c.char_start - 1] == "\n", (key, c.chunk_uid)
                if b.char_start < c.char_end < b.char_end:
                    assert text[c.char_end] == "\n", (key, c.chunk_uid)


def test_deterministic():
    _, a = chunks_for("bluewater")
    _, b = chunks_for("bluewater")
    assert [(c.chunk_uid, c.char_start, c.char_end) for c in a] == [
        (c.chunk_uid, c.char_start, c.char_end) for c in b
    ]
