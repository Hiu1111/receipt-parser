"""Tests for line item extraction."""

from decimal import Decimal

from receipt_parser.items import RowKind, classify_row, extract_items
from receipt_parser.layout import cluster_rows
from receipt_parser.models import BBox, Token
from receipt_parser.prices import find_price_column

PRICE_X = 350.0


def tok(text: str, x0: float, y0: float, w: float = 40, h: float = 12) -> Token:
    return Token(text=text, bbox=BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + h))


def priced(text: str, price: str, y: float) -> list[Token]:
    """A description on the left and a right-aligned price at PRICE_X."""
    width = len(price) * 10
    return [tok(text, 10, y), tok(price, PRICE_X - width, y, w=width)]


def build(rows_spec: list[list[Token]]):
    tokens = [t for row in rows_spec for t in row]
    rows = cluster_rows(tokens)
    return rows, find_price_column(rows)


def test_subtotal_is_not_read_as_total():
    """"TOTAL" is a substring of "SUBTOTAL"."""
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("SUBTOTAL", "12.99", 130),
    ])
    assert classify_row(rows[1], col, 20) is RowKind.SUMMARY


def test_priced_row_is_an_item():
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("FRIES", "4.50", 130),
    ])
    assert classify_row(rows[0], col, 20) is RowKind.ITEM


def test_unpriced_row_is_noise():
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("FRIES", "4.50", 130),
        [tok("THANK", 10, 160), tok("YOU", 60, 160)],
    ])
    assert classify_row(rows[2], col, 20) is RowKind.NOISE


def test_extracts_simple_items():
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("FRIES", "4.50", 130),
        priced("SODA", "3.25", 160),
    ])
    items, _ = extract_items(rows, col)
    assert [i.description for i in items] == ["BURGER", "FRIES", "SODA"]
    assert [i.total for i in items] == [
        Decimal("12.99"), Decimal("4.50"), Decimal("3.25"),
    ]


def test_stops_at_totals_block():
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("FRIES", "4.50", 130),
        priced("SUBTOTAL", "17.49", 160),
        priced("TAX", "1.40", 190),
        priced("TOTAL", "18.89", 220),
    ])
    items, summary = extract_items(rows, col)
    assert [i.description for i in items] == ["BURGER", "FRIES"]
    assert len(summary) == 3


def test_payment_lines_below_total_are_not_items():
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("TOTAL", "12.99", 130),
        priced("VISA", "12.99", 160),
    ])
    items, _ = extract_items(rows, col)
    assert len(items) == 1


def test_unpriced_row_after_summary_does_not_resume_items():
    rows, col = build([
        priced("BURGER", "12.99", 100),
        priced("TOTAL", "12.99", 130),
        priced("TIP", "3.00", 160),
        priced("MYSTERY", "5.00", 190),
    ])
    items, _ = extract_items(rows, col)
    assert [i.description for i in items] == ["BURGER"]


def test_at_notation():
    """"2 @ 4.99   9.98" is one item of quantity two."""
    tokens = [
        tok("2", 10, 100, w=10), tok("@", 30, 100, w=10),
        tok("4.99", 50, 100, w=40), tok("TACO", 110, 100),
        tok("9.98", 310, 100, w=40),
        *priced("SODA", "3.25", 130),
    ]
    rows = cluster_rows(tokens)
    col = find_price_column(rows)
    items, _ = extract_items(rows, col)

    taco = items[0]
    assert taco.quantity == 2
    assert taco.unit_price == Decimal("4.99")
    assert taco.total == Decimal("9.98")
    assert "TACO" in taco.description


def test_leading_quantity():
    tokens = [
        tok("3", 10, 100, w=10), tok("COOKIE", 30, 100),
        tok("6.00", 310, 100, w=40),
        *priced("SODA", "3.25", 130),
    ]
    rows = cluster_rows(tokens)
    items, _ = extract_items(rows, find_price_column(rows))
    assert items[0].quantity == 3
    assert items[0].description == "COOKIE"


def test_four_digit_leading_number_is_not_a_quantity():
    tokens = [
        tok("2024", 10, 100), tok("VINTAGE", 60, 100),
        tok("45.00", 305, 100, w=45),
        *priced("SODA", "3.25", 130),
    ]
    rows = cluster_rows(tokens)
    items, _ = extract_items(rows, find_price_column(rows))
    assert items[0].quantity == 1
    assert "2024" in items[0].description


def test_quantity_arithmetic_mismatch_lowers_confidence():
    """"3 @ 2.00" should extend to 6.00. If it says 9.00, something was misread."""
    tokens = [
        tok("3", 10, 100, w=10), tok("@", 30, 100, w=10),
        tok("2.00", 50, 100, w=40), tok("WATER", 110, 100),
        tok("9.00", 310, 100, w=40),
        *priced("SODA", "3.25", 130),
    ]
    rows = cluster_rows(tokens)
    items, _ = extract_items(rows, find_price_column(rows))
    assert items[0].total == Decimal("9.00")
    assert items[0].confidence <= 0.5


def test_ocr_repaired_price_lowers_confidence():
    rows, col = build([
        priced("BURGER", "I2.99", 100),
        priced("FRIES", "4.50", 130),
    ])
    items, _ = extract_items(rows, col)
    burger = next(i for i in items if i.description == "BURGER")
    assert burger.total == Decimal("12.99")
    assert burger.confidence < 1.0


def test_no_price_column_yields_no_items():
    rows, _ = build([priced("BURGER", "12.99", 100)])
    assert extract_items(rows, None) == ([], [])


def test_empty_rows():
    assert extract_items([], PRICE_X) == ([], [])


def test_price_with_no_description_is_skipped():
    tokens = [
        tok("12.99", 310, 100, w=40),
        *priced("SODA", "3.25", 130),
        *priced("WATER", "2.00", 160),
    ]
    rows = cluster_rows(tokens)
    items, _ = extract_items(rows, find_price_column(rows))
    assert [i.description for i in items] == ["SODA", "WATER"]
