"""Tests for price parsing and price-column detection."""

from decimal import Decimal

import pytest

from receipt_parser.layout import cluster_rows
from receipt_parser.models import BBox, Token
from receipt_parser.prices import find_price_column, parse_price


def tok(text: str, x0: float, y0: float, w: float = 40, h: float = 12) -> Token:
    return Token(text=text, bbox=BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + h))


@pytest.mark.parametrize("text,expected", [
    ("12.99", "12.99"),
    ("$12.99", "12.99"),
    ("  12.99  ", "12.99"),
    ("0.99", "0.99"),
    ("1,299.00", "1299.00"),
    ("12,99", "12.99"),
    ("1.299,50", "1299.50"),
    ("12. 99", "12.99"),
    ("12 .99", "12.99"),
])
def test_parses_valid_prices(text, expected):
    result = parse_price(text)
    assert result is not None, f"{text!r} should parse"
    assert result.value == Decimal(expected)
    assert result.repaired is False


@pytest.mark.parametrize("text,expected", [
    ("(3.50)", "-3.50"),
    ("-3.50", "-3.50"),
    ("3.50-", "-3.50"),
])
def test_parses_negative_prices(text, expected):
    result = parse_price(text)
    assert result is not None
    assert result.value == Decimal(expected)


@pytest.mark.parametrize("text", [
    "", "   ", "BURGER", "12.99.50", "12.999", "abc", "--", "$", "12/25",
])
def test_rejects_non_prices(text):
    assert parse_price(text) is None


def test_bare_integer_rejected_by_default():
    assert parse_price("2") is None
    assert parse_price("2024") is None


def test_bare_integer_accepted_when_caller_opts_in():
    result = parse_price("12", allow_bare_integer=True)
    assert result is not None
    assert result.value == Decimal("12")


@pytest.mark.parametrize("text,expected", [
    ("I2.99", "12.99"),
    ("l2.99", "12.99"),
    ("1O.50", "10.50"),
    ("S.99", "5.99"),
])
def test_repairs_ocr_digit_confusion(text, expected):
    result = parse_price(text)
    assert result is not None, f"{text!r} should parse after repair"
    assert result.value == Decimal(expected)
    assert result.repaired is True


def test_clean_parse_is_not_flagged_as_repaired():
    result = parse_price("12.99")
    assert result is not None and result.repaired is False


def test_repair_does_not_rescue_actual_words():
    """The repair table maps B->8, O->0, S->5, so "BOSS" becomes 8055.

    Repair is a last resort for tokens that are already mostly numeric;
    it must not turn arbitrary words into money.
    """
    assert parse_price("BOSS") is None
    assert parse_price("SOLD") is None


def _receipt_rows():
    tokens = [
        tok("BURGER", 10, 100), tok("12.99", 300, 100, w=50),
        tok("FRIES", 10, 130), tok("2", 200, 130, w=10), tok("4.50", 310, 130, w=40),
        tok("SODA", 10, 160), tok("3.25", 310, 160, w=40),
    ]
    return cluster_rows(tokens)


def test_finds_right_aligned_column():
    column = find_price_column(_receipt_rows())
    assert column is not None
    assert 345 <= column <= 355


def test_column_ignores_mid_row_quantity():
    column = find_price_column(_receipt_rows())
    assert column is not None and column > 300


def test_no_column_when_too_few_prices():
    tokens = [tok("BURGER", 10, 100), tok("12.99", 300, 100)]
    assert find_price_column(cluster_rows(tokens)) is None


def test_no_column_on_empty_input():
    assert find_price_column([]) is None


def test_left_aligned_prices_would_split_on_left_edges():
    """"9.50" and "112.00" share a right edge but start 30px apart."""
    tokens = [
        tok("A", 10, 100), tok("9.50", 320, 100, w=30),
        tok("B", 10, 130), tok("112.00", 290, 130, w=60),
    ]
    column = find_price_column(cluster_rows(tokens))
    assert column is not None
    assert 348 <= column <= 352
