"""Tests for row clustering.

Synthetic tokens, no OCR dependency. This is deliberate: the layout
algorithm is where most parsing bugs live, and being able to construct an
exact adversarial layout in three lines is worth more than testing against
real images at this stage.
"""

from receipt_parser.layout import cluster_rows, estimate_skew
from receipt_parser.models import BBox, Token


def tok(text: str, x0: float, y0: float, w: float = 40, h: float = 12) -> Token:
    return Token(text=text, bbox=BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + h))


def test_empty_input():
    assert cluster_rows([]) == []


def test_single_row():
    tokens = [tok("BURGER", 10, 100), tok("12.99", 300, 100)]
    rows = cluster_rows(tokens)
    assert len(rows) == 1
    assert rows[0].text == "BURGER 12.99"


def test_separates_vertically_distinct_rows():
    tokens = [tok("BURGER", 10, 100), tok("FRIES", 10, 130)]
    rows = cluster_rows(tokens)
    assert len(rows) == 2
    assert rows[0].text == "BURGER"
    assert rows[1].text == "FRIES"


def test_reading_order_is_ignored():
    """The price arrives before the description in OCR order.

    This is the case that breaks naive parsers: they emit the tokens in the
    order received and lose the description/price association.
    """
    tokens = [tok("12.99", 300, 100), tok("BURGER", 10, 102)]
    rows = cluster_rows(tokens)
    assert len(rows) == 1
    assert rows[0].text == "BURGER 12.99"  # re-sorted left to right


def test_slight_vertical_jitter_stays_one_row():
    """Real OCR boxes on the same line are never perfectly aligned."""
    tokens = [
        tok("CHKN", 10, 100),
        tok("SAND", 60, 103),
        tok("9.50", 300, 101),
    ]
    rows = cluster_rows(tokens)
    assert len(rows) == 1


def test_tall_token_does_not_swallow_short_ones():
    """A large merchant header spans several short lines vertically.

    Normalizing overlap by the *shorter* height keeps it from absorbing
    every token beneath it.
    """
    tokens = [
        tok("JOES", 10, 100, w=200, h=60),   # tall banner text
        tok("BURGER", 10, 200),
        tok("FRIES", 10, 230),
    ]
    rows = cluster_rows(tokens)
    assert len(rows) == 3


def test_tight_rows_do_not_merge():
    """Dense thermal receipts pack lines close together."""
    tokens = [
        tok("ITEM1", 10, 100, h=10),
        tok("ITEM2", 10, 111, h=10),  # 1px gap
    ]
    rows = cluster_rows(tokens)
    assert len(rows) == 2


def test_degenerate_zero_height_box():
    """OCR occasionally emits a zero-height box. Must not divide by zero."""
    tokens = [
        Token(text="X", bbox=BBox(x0=10, y0=100, x1=20, y1=100)),
        tok("Y", 40, 100),
    ]
    rows = cluster_rows(tokens)
    assert len(rows) == 1


def test_skew_detected_on_raw_tokens():
    """A rotated photo drifts downward left-to-right.

    Note this is measured on raw tokens, not rows: at this much drift the
    two tokens on each printed line no longer overlap vertically, so
    cluster_rows() correctly splits them and any row-based estimate would
    see nothing.
    """
    tokens = [
        tok("A", 0, 100),
        tok("B", 500, 110),   # 10px drift over 500px == 20 per 1000
        tok("C", 0, 200),
        tok("D", 500, 210),
    ]
    assert estimate_skew(tokens) > 15

    # and confirm the premise: clustering has already failed here
    assert len(cluster_rows(tokens)) == 4


def test_skew_zero_on_square_image():
    tokens = [tok("A", 0, 100), tok("B", 500, 100)]
    assert estimate_skew(tokens) == 0.0


def test_skew_ignores_cross_line_pairs():
    """Tokens on different printed lines must not register as drift."""
    tokens = [
        tok("A", 0, 100), tok("B", 500, 100),
        tok("C", 0, 130), tok("D", 500, 130),
    ]
    assert abs(estimate_skew(tokens)) < 1.0
