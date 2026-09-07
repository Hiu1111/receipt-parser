"""Tests for bill splitting.

The recurring assertion is that shares sum exactly to the amount being
split. Every rounding path has to preserve that, and it is the one bug a
user would notice immediately.
"""

from decimal import Decimal

import pytest

from receipt_parser.models import LineItem, ParsedReceipt, ParseStatus
from receipt_parser.split import Assignment, split_bill


def receipt(items, *, tax=None, tip=None, status=ParseStatus.OK):
    line_items = [
        LineItem(description=desc, total=Decimal(amount)) for desc, amount in items
    ]
    subtotal = sum((i.total for i in line_items), Decimal("0"))
    total = subtotal + Decimal(tax or "0") + Decimal(tip or "0")
    return ParsedReceipt(
        line_items=line_items,
        subtotal=subtotal,
        tax=Decimal(tax) if tax else None,
        tip=Decimal(tip) if tip else None,
        total=total,
        status=status,
    )


def totals(result):
    return {s.name: s.total for s in result.shares}


def test_each_person_pays_for_their_own_item():
    r = receipt([("BURGER", "10.00"), ("SALAD", "8.00")])
    result = split_bill(r, [
        Assignment(item_index=0, people=["alice"]),
        Assignment(item_index=1, people=["bob"]),
    ])
    assert totals(result) == {"alice": Decimal("10.00"), "bob": Decimal("8.00")}


def test_shared_item_splits_evenly():
    r = receipt([("PIZZA", "20.00")])
    result = split_bill(r, [Assignment(item_index=0, people=["alice", "bob"])])
    assert totals(result) == {"alice": Decimal("10.00"), "bob": Decimal("10.00")}


def test_person_can_have_multiple_items():
    r = receipt([("BURGER", "10.00"), ("FRIES", "4.00")])
    result = split_bill(r, [
        Assignment(item_index=0, people=["alice"]),
        Assignment(item_index=1, people=["alice"]),
    ])
    assert totals(result) == {"alice": Decimal("14.00")}


def test_uneven_split_sums_exactly():
    """$10 three ways is 3.333... Shares must still total 10.00."""
    r = receipt([("APPETIZER", "10.00")])
    result = split_bill(r, [Assignment(item_index=0, people=["a", "b", "c"])])
    assert result.total == Decimal("10.00")
    assert sorted(totals(result).values()) == [
        Decimal("3.33"), Decimal("3.33"), Decimal("3.34"),
    ]


@pytest.mark.parametrize("amount,parts", [
    ("10.00", 3), ("0.01", 2), ("0.05", 3), ("100.00", 7), ("19.99", 6),
])
def test_any_split_sums_exactly(amount, parts):
    r = receipt([("ITEM", amount)])
    people = [f"p{i}" for i in range(parts)]
    result = split_bill(r, [Assignment(item_index=0, people=people)])
    assert result.total == Decimal(amount)


def test_one_cent_among_two_people():
    """Someone gets the penny and someone gets nothing. Both are valid."""
    r = receipt([("MINT", "0.01")])
    result = split_bill(r, [Assignment(item_index=0, people=["a", "b"])])
    assert result.total == Decimal("0.01")


def test_tax_is_proportional_not_even():
    """The salad eater should not subsidize the steak eater."""
    r = receipt([("STEAK", "30.00"), ("SALAD", "10.00")], tax="4.00")
    result = split_bill(r, [
        Assignment(item_index=0, people=["alice"]),
        Assignment(item_index=1, people=["bob"]),
    ])
    by_name = {s.name: s for s in result.shares}
    assert by_name["alice"].tax == Decimal("3.00")
    assert by_name["bob"].tax == Decimal("1.00")


def test_tip_is_proportional():
    r = receipt([("STEAK", "30.00"), ("SALAD", "10.00")], tip="8.00")
    result = split_bill(r, [
        Assignment(item_index=0, people=["alice"]),
        Assignment(item_index=1, people=["bob"]),
    ])
    by_name = {s.name: s for s in result.shares}
    assert by_name["alice"].tip == Decimal("6.00")
    assert by_name["bob"].tip == Decimal("2.00")


def test_tax_allocation_sums_exactly_when_it_does_not_divide():
    r = receipt([("A", "10.00"), ("B", "10.00"), ("C", "10.00")], tax="1.00")
    result = split_bill(r, [
        Assignment(item_index=0, people=["a"]),
        Assignment(item_index=1, people=["b"]),
        Assignment(item_index=2, people=["c"]),
    ])
    assert sum((s.tax for s in result.shares), Decimal("0")) == Decimal("1.00")


def test_full_bill_reconciles_to_the_receipt_total():
    r = receipt(
        [("STEAK", "31.99"), ("SALAD", "11.50"), ("WINE", "17.00")],
        tax="4.83", tip="12.10",
    )
    result = split_bill(r, [
        Assignment(item_index=0, people=["alice"]),
        Assignment(item_index=1, people=["bob"]),
        Assignment(item_index=2, people=["alice", "bob", "carol"]),
    ])
    assert result.total == r.total


def test_no_tax_or_tip():
    r = receipt([("COFFEE", "3.50")])
    result = split_bill(r, [Assignment(item_index=0, people=["alice"])])
    assert result.shares[0].tax == Decimal("0")
    assert result.shares[0].tip == Decimal("0")


def test_unassigned_items_are_reported_not_hidden():
    """A forgotten item means someone is about to be undercharged."""
    r = receipt([("BURGER", "10.00"), ("FORGOTTEN", "8.00")])
    result = split_bill(r, [Assignment(item_index=0, people=["alice"])])
    assert result.unassigned_item_indices == [1]
    assert any("not assigned" in w for w in result.warnings)
    assert result.total == Decimal("10.00")


def test_no_assignments_yields_no_shares():
    r = receipt([("BURGER", "10.00")])
    result = split_bill(r, [])
    assert result.shares == []
    assert result.warnings


def test_out_of_range_item_index_is_ignored():
    r = receipt([("BURGER", "10.00")])
    result = split_bill(r, [
        Assignment(item_index=0, people=["alice"]),
        Assignment(item_index=99, people=["bob"]),
    ])
    assert totals(result)["alice"] == Decimal("10.00")
    assert totals(result)["bob"] == Decimal("0.00")


def test_unreconciled_receipt_warns_before_paying():
    r = receipt([("BURGER", "10.00")], status=ParseStatus.UNRECONCILED)
    result = split_bill(r, [Assignment(item_index=0, people=["alice"])])
    assert any("did not reconcile" in w for w in result.warnings)


def test_split_is_deterministic():
    """Two people comparing screens must see identical numbers."""
    r = receipt([("APPETIZER", "10.00")], tax="0.85")
    assignments = [Assignment(item_index=0, people=["a", "b", "c"])]
    assert totals(split_bill(r, assignments)) == totals(split_bill(r, assignments))
