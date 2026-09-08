"""Extract the totals block and check whether the parse adds up.

This is the module that decides whether to trust everything upstream.
Row clustering, price parsing, and item extraction are all heuristics, and
each can fail quietly -- a missed item, a misread digit, a description
absorbed into the wrong row. None of those announce themselves.

Arithmetic does. A receipt carries its own checksum: the items sum to the
subtotal, and subtotal plus tax plus tip equals the total. When those
don't hold, something upstream is wrong even if every individual value
looks plausible. That is the whole value of this module -- it converts
silent parse failures into loud ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from receipt_parser.items import _column_price, keyword_in_text  # noqa: F401
from receipt_parser.models import LineItem, ParsedReceipt, ParseStatus, Row
from receipt_parser.prices import parse_price

# Sub-cent drift is a rounding artifact, not a parse error. Two cents of
# slack absorbs tax rounding on a multi-item check without hiding a
# genuinely missing item, which is almost never worth less than a dollar.
TOLERANCE = Decimal("0.02")

# Order matters: SUBTOTAL must be tested before TOTAL.
_FIELD_KEYWORDS: list[tuple[str, str]] = [
    ("subtotal", "SUBTOTAL"),
    ("subtotal", "SUB TOTAL"),
    ("subtotal", "SUB-TOTAL"),
    ("tip", "GRATUITY"),
    ("tip", "TIP"),
    ("tax", "TAX"),
    ("total", "GRAND TOTAL"),
    ("total", "AMOUNT DUE"),
    ("total", "BALANCE DUE"),
    ("total", "TOTAL"),
]

# Payment tender lines. They repeat the total and must not be read as one.
_PAYMENT_KEYWORDS = [
    "CHANGE", "CASH", "CREDIT", "DEBIT",
    "VISA", "MASTERCARD", "AMEX", "DISCOVER",
]


@dataclass
class Totals:
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    tip: Decimal | None = None
    total: Decimal | None = None


def extract_totals(
    rows: list[Row],
    summary_indices: list[int],
    price_column: float | None,
    tolerance: float,
) -> Totals:
    """Read subtotal, tax, tip and total out of the summary rows.

    Multiple tax lines are summed. State and city tax are frequently
    printed separately, and treating the last one as "the" tax would
    understate it.
    """
    totals = Totals()
    tax_parts: list[Decimal] = []

    if price_column is None:
        return totals

    for index in summary_indices:
        row = rows[index]
        text = row.text.upper()

        if any(keyword_in_text(text, word) for word in _PAYMENT_KEYWORDS):
            continue

        field = _field_for(text)
        if field is None:
            continue

        amount = _column_price(row, price_column, tolerance)
        if amount is None:
            continue

        if field == "tax":
            tax_parts.append(amount.value)
        elif field == "subtotal" and totals.subtotal is None:
            totals.subtotal = amount.value
        elif field == "tip" and totals.tip is None:
            totals.tip = amount.value
        elif field == "total" and totals.total is None:
            totals.total = amount.value

    if tax_parts:
        totals.tax = sum(tax_parts, Decimal("0"))

    return totals


def _field_for(text: str) -> str | None:
    for field, keyword in _FIELD_KEYWORDS:
        if keyword_in_text(text, keyword):
            return field
    return None


def reconcile(
    items: list[LineItem],
    totals: Totals,
) -> tuple[ParseStatus, list[str]]:
    """Check the parse against the receipt's own arithmetic.

    Returns a status and human-readable warnings. Warnings name the
    discrepancy in dollars rather than saying "parse failed", because the
    size of the gap tells a reviewer where to look: a gap equal to one
    item's price means a row was missed, while a few cents means a digit
    was misread.
    """
    warnings: list[str] = []

    if totals.total is None:
        warnings.append("No total found on the receipt.")
        return ParseStatus.FAILED, warnings

    if not items:
        warnings.append("No line items were extracted.")
        return ParseStatus.UNRECONCILED, warnings

    items_sum = sum((i.total for i in items), Decimal("0"))

    # Check 1: items against the printed subtotal. This is the check that
    # catches a dropped or duplicated item, and it is the most valuable
    # one -- a missing item is invisible in every other way.
    if totals.subtotal is not None:
        gap = items_sum - totals.subtotal
        if abs(gap) > TOLERANCE:
            warnings.append(
                f"Line items total {items_sum} but the subtotal is "
                f"{totals.subtotal} (off by {abs(gap)})."
            )

    # Check 2: the totals block against itself.
    base = totals.subtotal if totals.subtotal is not None else items_sum
    computed = base + (totals.tax or Decimal("0")) + (totals.tip or Decimal("0"))
    gap = computed - totals.total
    if abs(gap) > TOLERANCE:
        warnings.append(
            f"Subtotal plus tax and tip is {computed} but the total is "
            f"{totals.total} (off by {abs(gap)})."
        )

    # Low-confidence items are not an arithmetic failure, but they are
    # worth surfacing for review alongside one.
    uncertain = [i for i in items if i.confidence < 1.0]
    if uncertain:
        names = ", ".join(i.description for i in uncertain[:3])
        warnings.append(f"{len(uncertain)} item(s) need review: {names}.")

    if any("off by" in w for w in warnings):
        return ParseStatus.UNRECONCILED, warnings

    return ParseStatus.OK, warnings


def build_receipt(
    rows: list[Row],
    items: list[LineItem],
    summary_indices: list[int],
    price_column: float | None,
    *,
    tolerance: float = 20.0,
) -> ParsedReceipt:
    """Assemble the final result.

    Note the ordering: status is computed before construction, because
    ParsedReceipt refuses to be built with status=OK and no total. The
    validator is not decoration -- it is what makes it impossible to
    return a confident answer that was never checked.
    """
    totals = extract_totals(rows, summary_indices, price_column, tolerance)
    status, warnings = reconcile(items, totals)

    return ParsedReceipt(
        merchant=_merchant(rows),
        line_items=items,
        subtotal=totals.subtotal,
        tax=totals.tax,
        tip=totals.tip,
        total=totals.total,
        status=status,
        warnings=warnings,
    )


def _merchant(rows: list[Row]) -> str | None:
    """Best guess at the merchant name: the first substantive header row.

    Receipts put the merchant at the top, above any prices. Rows that are
    mostly digits are phone numbers, dates, or register IDs.
    """
    for row in rows[:4]:
        text = row.text.strip()
        if len(text) < 3:
            continue
        if parse_price(text, allow_bare_integer=True) is not None:
            continue
        digits = sum(ch.isdigit() for ch in text)
        if digits > len(text) / 2:
            continue
        if re.search(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}", text):  # phone number
            continue
        return text
    return None
