"""Small text helpers shared by the parser, citation checks, and offline test doubles."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"[\s  -​  　]+")
_TRANS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)

STOPWORDS = frozenset(
    """a an and are as at be been but by did do does for from had has have how in into is it
    its of on or our over per that the their them then there these they this those to under
    up was we were what when where which while who whom why will with would you your about
    after all also any can could during each if may more most no not only other out same
    should so such than through very company company's companies report reported
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def normalize_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


_NO_SPACE_AFTER = set("$€£(")
_NO_SPACE_BEFORE = set(")%,;:")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize case, quotes, dashes and whitespace; return text plus a map from each
    normalized character back to its index in the original string."""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        c = unicodedata.normalize("NFKC", ch).translate(_TRANS).lower()
        for cc in c:
            if _WS.fullmatch(cc):
                if not chars or chars[-1] == " ":
                    continue
                cc = " "
            chars.append(cc)
            index_map.append(i)
    # Drop spaces that only differ by table formatting: "$ 1,234" == "$1,234", "(5 )" == "(5)".
    out_c: list[str] = []
    out_m: list[int] = []
    n = len(chars)
    for j, cc in enumerate(chars):
        if cc == " ":
            prev = out_c[-1] if out_c else ""
            nxt = chars[j + 1] if j + 1 < n else ""
            if not prev or not nxt or prev in _NO_SPACE_AFTER or nxt in _NO_SPACE_BEFORE:
                continue
        out_c.append(cc)
        out_m.append(index_map[j])
    return "".join(out_c), out_m


def norm_for_match(s: str) -> str:
    """Case/quote/dash/whitespace-insensitive form used to compare quotes with passages."""
    return _normalize_with_map(s)[0]


def tokens(s: str) -> list[str]:
    return _TOKEN.findall(norm_for_match(s))


def light_stem(tok: str) -> str:
    """Very small suffix stripper so 'revenues' matches 'revenue' in offline doubles."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"  # liabilities -> liability
    if tok.endswith(("sses", "xes", "zes", "ches", "shes")):
        return tok[:-2]  # losses -> loss, taxes -> tax
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]  # revenues -> revenue
    return tok


def content_tokens(s: str) -> list[str]:
    return [light_stem(t) for t in tokens(s) if t not in STOPWORDS]


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find ``quote`` in ``text`` tolerating whitespace, case, quote and dash variants.

    Returns (start, end) offsets into the original ``text`` or None.
    """
    quote = quote.strip()
    if not quote:
        return None
    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)
    norm_quote = norm_for_match(quote)
    if not norm_quote:
        return None
    norm_text, index_map = _normalize_with_map(text)
    pos = norm_text.find(norm_quote)
    if pos < 0:
        return None
    start = index_map[pos]
    end = index_map[pos + len(norm_quote) - 1] + 1
    return start, end


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"“$])|\n+")
_ABBREV = re.compile(
    r"(?:\b(?:Inc|Corp|Co|Ltd|No|Nos|U\.S|vs|e\.g|i\.e|approx|Mr|Ms|Dr|St|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)\.)$"
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence-ish spans (offsets), avoiding common abbreviations."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        piece = text[start : m.start()]
        if _ABBREV.search(piece):
            continue
        if piece.strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    # trim whitespace inside each span
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out
