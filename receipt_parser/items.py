"""Turn classified rows into line items and a totals block.

A receipt is three regions stacked vertically: header noise (merchant
address, date, register number), the item block, and the totals block
(subtotal, tax, tip, total, payment method). Only the middle region
produces line items, and the boundary between regions is not marked --
it has to be inferred.

The rule used here: the totals block begins at the first row matching a
summary keyword, and everything from there down is summary.
"""

from __future__ import annotations

import difflib
import re
from decimal import Decimal
from enum import Enum

from receipt_parser.models import LineItem, Row
from receipt_parser.prices import parse_price

# How close a token's right edge must be to the price column, as a
# fraction of page width, to count as sitting in that column.
COLUMN_TOLERANCE_RATIO = 0.04


# Below this length a keyword is too short to fuzzy-match safely: "TAX"
# is one edit from "TAG", "FAX" and "MAX", and a false summary row
# truncates the item list at whatever row it hits.
MIN_FUZZY_KEYWORD_LEN = 5

# One substitution in an eight-letter word scores about 0.93, a dropped
# character about 0.94. Two edits fall below this.
FUZZY_CUTOFF = 0.88


def keyword_in_text(text: str, keyword: str) -> bool:
    """Whether a keyword appears in a row, allowing for OCR damage.

    Exact substring first, which handles multi-word keywords and costs
    nothing. Only single long keywords fall through to fuzzy comparison.

    This exists because OCR splits words as readily as it misreads them.
    On a real Wendy's receipt "Subtotal" came back as "Subtota" and a
    separate "|", so an exact match failed and the subtotal row was
    classified as a line item -- which then broke reconciliation, since
    the subtotal was being counted as something somebody ate.
    """
    upper = text.upper()
    if keyword in upper:
        return True

    if len(keyword) < MIN_FUZZY_KEYWORD_LEN or " " in keyword:
        return False

    # Compare against alphabetic runs only. Punctuation that OCR invents
    # ("|", ".", ":") should not count against the similarity score.
    words = re.findall(r"[A-Z]+", upper)
    return bool(
        difflib.get_close_matches(keyword, words, n=1, cutoff=FUZZY_CUTOFF)
    )


class RowKind(str, Enum):
    ITEM = "item"
    SUMMARY = "summary"
    NOISE = "noise"


# Ordered longest-first: "SUBTOTAL" must be tested before "TOTAL", or every
# subtotal row is misread as the grand total.
_SUMMARY_KEYWORDS = [
    "SUBTOTAL", "SUB TOTAL", "SUB-TOTAL",
    "AMOUNT DUE", "BALANCE DUE", "GRAND TOTAL",
    "GRATUITY", "SERVICE CHARGE",
    "MASTERCARD", "AMEX", "DISCOVER",
    "CHANGE", "CREDIT", "DEBIT", "VISA", "CASH",
    "TOTAL", "TAX", "TIP",
]

# "2 @ 4.99" or "2 X 4.99" -- quantity and unit price, with the extended
# amount in the price column.
_QTY_AT_UNIT = re.compile(r"^(\d+)\s*[@xX]\s*([\d.,]+)\s*(.*)$")

# "2x Lorem ipsum" -- quantity glued to an x, then the description. Common
# on printed retail receipts. Distinct from _QTY_AT_UNIT because what
# follows the separator is a description, not a unit price, and there is
# no space to anchor on.
_QTY_X_DESC = re.compile(r"^(\d+)\s*[@xX]\s*(\D.*)$")

# "2 BURGER" -- leading quantity, no unit price given.
_LEADING_QTY = re.compile(r"^(\d+)\s+(\D.*)$")

# A lone currency symbol is its own OCR token on receipts that column-align
# the symbol separately from the amount. It sits left of the price column
# and would otherwise be swept into the description.
_CURRENCY_ONLY = re.compile(r"^[$\u20ac\u00a3\u00a5]+$")

# A UPC or SKU printed between the item name and the price. Eight digits
# is well past any quantity, weight, or size that belongs in a name.
_PRODUCT_CODE = re.compile(r"^\d{8,}$")

# Tax-status flags printed alongside the product code on grocery receipts.
_TAX_FLAG = re.compile(r"^[A-Z]$")

# Package-size units. "12 CT NITRIL" is one box of twelve, not twelve
# boxes -- the leading number sizes the package rather than counting it.
_PACKAGE_UNITS = {
    "CT", "PK", "PC", "PCS", "OZ", "LB", "LBS", "G", "KG",
    "ML", "L", "GAL", "QT", "PT", "IN", "FT", "MG",
}


def classify_row(row: Row, price_column: float | None, tolerance: float) -> RowKind:
    """Decide what a row is.

    Summary detection runs on text before price detection, because a
    totals row looks structurally identical to an item row -- description
    on the left, amount in the price column. Only the words distinguish them.
    """
    for keyword in _SUMMARY_KEYWORDS:
        if keyword_in_text(row.text, keyword):
            return RowKind.SUMMARY

    if price_column is None:
        return RowKind.NOISE
    if _column_price(row, price_column, tolerance) is None:
        return RowKind.NOISE

    return RowKind.ITEM


def extract_items(
    rows: list[Row],
    price_column: float | None,
    *,
    page_width: float | None = None,
) -> tuple[list[LineItem], list[int]]:
    """Extract line items, and the indices of rows classified as summary.

    Stops at the first summary row. A priced row below the subtotal is a
    payment line or a tax breakdown, not something anyone ate.
    """
    if price_column is None or not rows:
        return [], []

    if page_width is None:
        page_width = max(r.bbox.x1 for r in rows) - min(r.bbox.x0 for r in rows)
    tolerance = max(page_width * COLUMN_TOLERANCE_RATIO, 1.0)

    items: list[LineItem] = []
    summary_indices: list[int] = []
    seen_summary = False

    for index, row in enumerate(rows):
        kind = classify_row(row, price_column, tolerance)

        if kind is RowKind.SUMMARY:
            seen_summary = True
            summary_indices.append(index)
            continue

        # Once the totals block starts, nothing below it is an item --
        # even if it parses like one.
        if seen_summary:
            continue

        if kind is RowKind.ITEM:
            item = _build_item(row, index, price_column, tolerance)
            if item is not None:
                items.append(item)

    return items, summary_indices


def _build_item(
    row: Row,
    index: int,
    price_column: float,
    tolerance: float,
) -> LineItem | None:
    total = _column_price(row, price_column, tolerance)
    if total is None:
        return None

    description = _description_text(row, price_column, tolerance)
    if not description:
        return None

    quantity, unit_price, description = _split_quantity(description)

    confidence = 1.0
    if total.repaired:
        # OCR character substitution was needed. The value is a guess.
        confidence = 0.6

    if unit_price is not None and quantity > 0:
        expected = unit_price * quantity
        if expected != total.value:
            # "3 @ 2.00" that does not equal the extended amount means one
            # of the three numbers was misread. Keep the row but flag it --
            # dropping it would silently lose an item the diner ordered.
            confidence = min(confidence, 0.5)

    return LineItem(
        description=description,
        quantity=quantity,
        unit_price=unit_price,
        total=total.value,
        source_row_indices=[index],
        confidence=confidence,
    )


def _column_price(row: Row, price_column: float, tolerance: float):
    """The price sitting in the price column, if any.

    Scans right to left and takes the first match: on a row like
    "2 @ 4.99   9.98" both numbers may fall near the column, and the
    rightmost is the extended amount.

    Bare integers are allowed here specifically because position resolves
    the ambiguity that made them unsafe in general -- a lone "12" in the
    price column is $12.00, not a quantity.
    """
    for token in reversed(row.tokens):
        if abs(token.bbox.x1 - price_column) > tolerance:
            continue
        parsed = parse_price(token.text, allow_bare_integer=True)
        if parsed is not None:
            return parsed
    return None


def _description_text(row: Row, price_column: float, tolerance: float) -> str:
    """Everything left of the price column, cleaned.

    Three kinds of token are dropped:

    Standalone currency symbols, on receipts that column-align the symbol
    separately from the amount.

    Product codes. Grocery receipts print the UPC between the item name
    and the price ("BREAD  007225003712  F  2.88"), and a 12-digit number
    is never part of a product name.

    Single-letter tax flags, but only on rows that also carried a product
    code. Those flags sit in the same column block as the UPC, and tying
    the rule to the UPC keeps it from eating the "D" in "VITAMIN D" on
    receipts that print no codes at all.
    """
    candidates = [
        token.text
        for token in row.tokens
        if token.bbox.x1 < price_column - tolerance
        and not _CURRENCY_ONLY.match(token.text.strip())
    ]

    kept: list[str] = []
    saw_product_code = False
    for text in candidates:
        stripped = text.strip()
        if _PRODUCT_CODE.match(stripped):
            saw_product_code = True
            continue
        if saw_product_code and _TAX_FLAG.match(stripped):
            continue
        kept.append(text)

    return " ".join(kept).strip()

def _split_quantity(description: str) -> tuple[int, Decimal | None, str]:
    """Pull a leading quantity and optional unit price out of a description.

    Returns (quantity, unit_price, remaining_description).
    """
    match = _QTY_AT_UNIT.match(description)
    if match:
        qty_text, unit_text, rest = match.groups()
        unit = parse_price(unit_text)
        return int(qty_text), (unit.value if unit else None), rest.strip()

    # Tried after _QTY_AT_UNIT, because "2 @ 4.99 TACO" matches both and
    # only the first recovers the unit price.
    match = _QTY_X_DESC.match(description)
    if match:
        qty_text, rest = match.groups()
        quantity = int(qty_text)
        if quantity > 999:
            return 1, None, description
        return quantity, None, rest.strip()

    match = _LEADING_QTY.match(description)
    if match:
        qty_text, rest = match.groups()
        quantity = int(qty_text)
        # "12 CT NITRIL" is one twelve-count box, not twelve boxes.
        first_word = rest.split()[0].upper().rstrip(".") if rest.split() else ""
        if first_word in _PACKAGE_UNITS:
            return 1, None, description
        # A four-digit leading number is a year or a SKU, not a quantity.
        # Nobody orders 2024 burgers.
        if quantity > 999:
            return 1, None, description
        return quantity, None, rest.strip()

    return 1, None, description
