"""Tests for totals extraction and arithmetic reconciliation."""

from decimal import Decimal

import pytest

from receipt_parser.items import extract_items
from receipt_parser.layout import cluster_rows
from receipt_parser.models import BBox, LineItem, ParsedReceipt, ParseStatus, Token
from receipt_parser.prices import find_price_column
from receipt_parser.reconcile import Totals, build_receipt, extract_totals, reconcile

PRICE_X = 350.0


def tok(text: str, x0: float, y0: float, w: float = 40, h: float = 12) -> Token:
    return Token(text=text, bbox=BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + h))


def priced(text: str, price: str, y: float) -> list[Token]:
    width = len(price) * 10
    return [tok(text, 10, y), tok(price, PRICE_X - width, y, w=width)]


def parse(rows_spec):
    tokens = [t for row in rows_spec for t in row]
    rows = cluster_rows(tokens)
    column = find_price_column(rows)
    items, summary = extract_items(rows, column)
    return build_receipt(rows, items, summary, column)


def item(desc: str, total: str, confidence: float = 1.0) -> LineItem:
    return LineItem(description=desc, total=Decimal(total), confidence=confidence)


def _totals_from(rows_spec):
    rows = cluster_rows([t for r in rows_spec for t in r])
    column = find_price_column(rows)
    _, summary = extract_items(rows, column)
    return extract_totals(rows, summary, column, 20.0)


def test_extracts_all_four_fields():
    totals = _totals_from([
        priced("BURGER", "10.00", 100),
        priced("SUBTOTAL", "10.00", 130),
        priced("TAX", "0.80", 160),
        priced("TIP", "2.00", 190),
        priced("TOTAL", "12.80", 220),
    ])
    assert totals.subtotal == Decimal("10.00")
    assert totals.tax == Decimal("0.80")
    assert totals.tip == Decimal("2.00")
    assert totals.total == Decimal("12.80")


def test_multiple_tax_lines_are_summed():
    """State and city tax are often printed separately."""
    totals = _totals_from([
        priced("BURGER", "10.00", 100),
        priced("SUBTOTAL", "10.00", 130),
        priced("STATE TAX", "0.60", 160),
        priced("CITY TAX", "0.25", 190),
        priced("TOTAL", "10.85", 220),
    ])
    assert totals.tax == Decimal("0.85")


def test_payment_line_is_not_read_as_total():
    """A VISA line repeats the total amount and must be ignored."""
    totals = _totals_from([
        priced("BURGER", "10.00", 100),
        priced("TOTAL", "10.00", 130),
        priced("VISA", "10.00", 160),
        priced("CHANGE", "0.00", 190),
    ])
    assert totals.total == Decimal("10.00")


def test_gratuity_is_read_as_tip():
    totals = _totals_from([
        priced("BURGER", "10.00", 100),
        priced("GRATUITY", "2.00", 130),
        priced("TOTAL", "12.00", 160),
    ])
    assert totals.tip == Decimal("2.00")


def test_clean_receipt_reconciles():
    items = [item("BURGER", "10.00"), item("FRIES", "5.00")]
    totals = Totals(subtotal=Decimal("15.00"), tax=Decimal("1.20"),
                    tip=Decimal("3.00"), total=Decimal("19.20"))
    status, warnings = reconcile(items, totals)
    assert status is ParseStatus.OK
    assert warnings == []


def test_missing_item_is_caught():
    """The check that matters: a dropped row is invisible any other way."""
    items = [item("BURGER", "10.00")]  # FRIES was missed
    totals = Totals(subtotal=Decimal("15.00"), total=Decimal("15.00"))
    status, warnings = reconcile(items, totals)
    assert status is ParseStatus.UNRECONCILED
    assert any("5.00" in w for w in warnings)


def test_totals_block_inconsistent_with_itself():
    items = [item("BURGER", "10.00")]
    totals = Totals(subtotal=Decimal("10.00"), tax=Decimal("0.80"),
                    total=Decimal("99.99"))
    status, _ = reconcile(items, totals)
    assert status is ParseStatus.UNRECONCILED


def test_penny_rounding_is_tolerated():
    """Tax rounding produces sub-cent drift that is not a parse error."""
    items = [item("BURGER", "10.00")]
    totals = Totals(subtotal=Decimal("10.00"), tax=Decimal("0.83"),
                    total=Decimal("10.84"))
    status, _ = reconcile(items, totals)
    assert status is ParseStatus.OK


def test_no_total_is_a_failure():
    status, warnings = reconcile([item("BURGER", "10.00")], Totals())
    assert status is ParseStatus.FAILED
    assert any("total" in w.lower() for w in warnings)


def test_no_items_is_unreconciled():
    status, _ = reconcile([], Totals(total=Decimal("10.00")))
    assert status is ParseStatus.UNRECONCILED


def test_missing_subtotal_falls_back_to_items_sum():
    """Fast-food receipts often print no subtotal line."""
    items = [item("BURGER", "10.00"), item("FRIES", "5.00")]
    totals = Totals(tax=Decimal("1.20"), total=Decimal("16.20"))
    status, _ = reconcile(items, totals)
    assert status is ParseStatus.OK


def test_low_confidence_items_are_surfaced():
    items = [item("BURGER", "10.00", confidence=0.6)]
    totals = Totals(subtotal=Decimal("10.00"), total=Decimal("10.00"))
    status, warnings = reconcile(items, totals)
    assert status is ParseStatus.OK
    assert any("review" in w for w in warnings)


def test_full_pipeline_on_a_clean_receipt():
    receipt = parse([
        [tok("JOES", 10, 40), tok("DINER", 60, 40)],
        priced("BURGER", "10.00", 100),
        priced("FRIES", "5.00", 130),
        priced("SUBTOTAL", "15.00", 160),
        priced("TAX", "1.20", 190),
        priced("TOTAL", "16.20", 220),
    ])
    assert receipt.status is ParseStatus.OK
    assert receipt.merchant == "JOES DINER"
    assert len(receipt.line_items) == 2
    assert receipt.total == Decimal("16.20")


def test_full_pipeline_flags_a_missed_item():
    """The FRIES price is unparseable, so the item is dropped upstream.

    Reconciliation is what notices. Without it the receipt would come back
    looking complete and simply be wrong.
    """
    receipt = parse([
        priced("BURGER", "10.00", 100),
        [tok("FRIES", 10, 130), tok("~~~~", 310, 130, w=40)],
        priced("SUBTOTAL", "15.00", 160),
        priced("TOTAL", "15.00", 190),
    ])
    assert receipt.status is ParseStatus.UNRECONCILED
    assert any("off by" in w for w in receipt.warnings)


def test_ok_status_requires_a_total_at_the_type_level():
    """Even if reconcile() had a bug, the model refuses to be built."""
    with pytest.raises(ValueError):
        ParsedReceipt(status=ParseStatus.OK, total=None)
