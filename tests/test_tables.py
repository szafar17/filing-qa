from __future__ import annotations

from lxml import html as lxml_html

from filingqa.ingest.tables import Cell, flatten_table, is_numeric_value, is_value, merge_fragments


def table(markup: str):
    return lxml_html.fragment_fromstring(markup)


SEC_STYLE = """<table>
<tr><td></td><td colspan="3">Year Ended December 31,</td><td></td><td></td></tr>
<tr><td></td><td>2024</td><td></td><td>2023</td><td></td><td>% Change</td></tr>
<tr><td>Industrial Automation</td><td>$</td><td>812.9</td><td>$</td><td>741.0</td><td>9.7%</td></tr>
<tr><td>Warehouse Systems</td><td></td><td>471.7</td><td></td><td>389.5</td><td>21.1%</td></tr>
<tr><td>Interest expense</td><td></td><td>(6.2</td><td>)</td><td>(9.8</td><td>)</td></tr>
</table>"""


def test_value_detection():
    for v in ["1,234", "$1,234.5", "(6.2)", "$(3.4)", "12.5%", "(1.8)%", "—", "-", "n/a"]:
        assert is_value(v), v
    assert is_value("2024") and not is_numeric_value("2024")  # bare year = header, not data
    assert not is_value("Revenue")


def test_currency_and_paren_fragments_are_merged():
    cells = [Cell("Net loss", 0), Cell("$", 1), Cell("(3.4", 2), Cell(")", 3), Cell("", 4)]
    merged = merge_fragments(cells)
    assert [c.text for c in merged] == ["Net loss", "$(3.4)"]
    # the merged value spans from the "$" cell through the number cell
    assert (merged[1].col, merged[1].span) == (1, 2)


def test_sec_style_table_rows_keep_label_value_and_header():
    flat = flatten_table(table(SEC_STYLE))
    assert flat.is_data
    assert flat.header_line == (
        "Columns: Year Ended December 31, 2024 | Year Ended December 31, 2023 | % Change"
    )
    assert flat.lines[1] == (
        "Industrial Automation | Year Ended December 31, 2024: $812.9 | "
        "Year Ended December 31, 2023: $741.0 | % Change: 9.7%"
    )
    # a value without "$" whose year header sits one column to the left still gets that year
    assert flat.lines[2].startswith("Warehouse Systems | Year Ended December 31, 2024: 471.7")
    assert "Year Ended December 31, 2023: (9.8)" in flat.lines[3]


def test_multilevel_headers():
    t = """<table>
    <tr><td></td><td colspan="2">Three Months Ended June 30,</td><td></td>
        <td colspan="2">Six Months Ended June 30,</td></tr>
    <tr><td></td><td>2024</td><td>2023</td><td></td><td>2024</td><td>2023</td></tr>
    <tr><td>Revenue</td><td>10.0</td><td>8.0</td><td></td><td>19.0</td><td>15.0</td></tr>
    <tr><td>Net income</td><td>1.0</td><td>0.5</td><td></td><td>2.0</td><td>1.1</td></tr>
    </table>"""
    flat = flatten_table(table(t))
    assert flat.lines[1] == (
        "Revenue | Three Months Ended June 30, 2024: 10.0 | Three Months Ended June 30, 2023: 8.0 | "
        "Six Months Ended June 30, 2024: 19.0 | Six Months Ended June 30, 2023: 15.0"
    )


def test_wide_header_does_not_leak_into_neighbor_column():
    t = """<table>
    <tr><td></td><td colspan="2">Fiscal Year Ended June 30,</td><td></td></tr>
    <tr><td></td><td>2024</td><td>2023</td><td>Change</td></tr>
    <tr><td>Retail</td><td>$ 1,301.2</td><td>$ 1,214.9</td><td>7.1%</td></tr>
    <tr><td>Foodservice</td><td>541.3</td><td>551.1</td><td>(1.8)%</td></tr>
    </table>"""
    flat = flatten_table(table(t))
    assert flat.header_line == (
        "Columns: Fiscal Year Ended June 30, 2024 | Fiscal Year Ended June 30, 2023 | Change"
    )


def test_units_caption_and_label_rows():
    t = """<table>
    <tr><td>(In millions)</td><td>2024</td><td>2023</td></tr>
    <tr><td colspan="3">Operating expenses:</td></tr>
    <tr><td>Research</td><td>96.3</td><td>84.7</td></tr>
    <tr><td>Selling</td><td>214.5</td><td>198.2</td></tr>
    </table>"""
    flat = flatten_table(table(t))
    assert flat.lines[0] == "(In millions)"
    assert "Operating expenses:" in flat.lines
    assert "Research | 2024: 96.3 | 2023: 84.7" in flat.lines


def test_new_header_block_resets_columns():
    t = """<table>
    <tr><td></td><td>2024</td><td>2023</td></tr>
    <tr><td>Assets</td><td>5</td><td>4</td></tr>
    <tr><td>Equity</td><td>3</td><td>2</td></tr>
    <tr><td></td><td>Q4</td><td>Q3</td></tr>
    <tr><td>Revenue</td><td>9</td><td>8</td></tr>
    <tr><td>Costs</td><td>7</td><td>6</td></tr>
    </table>"""
    flat = flatten_table(table(t))
    assert "Revenue | Q4: 9 | Q3: 8" in flat.lines
    assert "Assets | 2024: 5 | 2023: 4" in flat.lines


def test_layout_tables_become_text_rows():
    flat = flatten_table(table("<table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>"))
    assert not flat.is_data
    assert flat.layout_rows == [("Item 1A. Risk Factors", False)]


def test_toc_rows_flagged():
    t = """<table>
    <tr><td><a href="#i1">Item 1.</a></td><td>Business</td></tr>
    <tr><td>Item 7.</td><td>MD&amp;A</td><td>12</td></tr>
    </table>"""
    flat = flatten_table(table(t))
    assert not flat.is_data
    assert [toc for _, toc in flat.layout_rows] == [True, True]
