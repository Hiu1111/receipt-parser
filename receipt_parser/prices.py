"""Turn OCR tokens into money, and find where money lives on the page.

Two separate problems that look like one:

1. Is this string a price, and if so what number is it? Currency symbols,
   European decimal commas, thousands separators, parenthesized discounts,
   and OCR reading digits as letters all land here.

2. Which x-position holds the prices? Receipts right-align prices in a
   column. Knowing that column is what distinguishes a line total from the
   quantity in "2 @ 4.99" -- both are numbers on the same row, and only
   position tells them apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from receipt_parser.models import Row

# Characters OCR commonly emits in place of digits on thermal paper.
# Applied only as a fallback, and only when the repair makes an otherwise
# unparseable token parse cleanly.
_OCR_DIGIT_REPAIRS = str.maketrans({
    "I": "1", "l": "1", "|": "1",
    "O": "0", "o": "0", "D": "0",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
})

_CURRENCY = "$\u20ac\u00a3\u00a5"

# A price is digits, optional group separators, optional decimal part.
# Anchored: partial matches inside a longer token are not prices.
_PRICE_SHAPE = re.compile(r"^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?$|^\d+(?:[.,]\d{1,2})?$")


@dataclass(frozen=True)
class PriceParse:
    value: Decimal
    # True when OCR character substitution was needed to parse this.
    # Downstream confidence scoring should discount repaired values -- the
    # substitution table is a guess, not a certainty.
    repaired: bool = False


def parse_price(text: str, *, allow_bare_integer: bool = False) -> PriceParse | None:
    """Parse a token as a monetary amount, or return None.

    Returning None is a real answer, not a failure to try. A token that is
    not confidently a price must not be coerced into one -- a wrong number
    that looks right is worse than an admitted gap, because nothing
    downstream can detect it.

    `allow_bare_integer` exists because "12" is genuinely ambiguous. In the
    price column it is $12.00; anywhere else it is far more likely a
    quantity, a SKU fragment, or part of a date. Callers that know the
    token's position decide.
    """
    cleaned = _clean(text)
    if not cleaned:
        return None

    negative = False
    # Parenthesized amounts are discounts/credits on most POS systems.
    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:].strip()
    elif cleaned.endswith("-"):
        # Trailing minus: an AS/400-era convention still common on receipts.
        negative = True
        cleaned = cleaned[:-1].strip()

    result = _to_decimal(cleaned, repaired=False)
    effective = cleaned
    if result is None:
        # Repair is only for tokens that are already mostly a number. A
        # token with no digits at all is a word, and the substitution table
        # will happily turn "BOSS" into 8055 if allowed to try.
        if not any(ch.isdigit() for ch in cleaned):
            return None
        repaired_text = cleaned.translate(_OCR_DIGIT_REPAIRS)
        if repaired_text == cleaned:
            return None
        result = _to_decimal(repaired_text, repaired=True)
        if result is None:
            return None
        effective = repaired_text

    # Checked against the repaired text, not the original: "I2" repairs to
    # "12", which is a bare integer and subject to the same ambiguity.
    if not allow_bare_integer and _is_bare_integer(effective):
        return None

    if negative:
        result = PriceParse(value=-result.value, repaired=result.repaired)
    return result


def _clean(text: str) -> str:
    out = text.strip()
    for symbol in _CURRENCY:
        out = out.replace(symbol, "")
    # OCR frequently splits "12.99" into "12. 99" or "12 .99".
    out = re.sub(r"\s*([.,])\s*", r"\1", out)
    return out.strip()


def _to_decimal(text: str, *, repaired: bool) -> PriceParse | None:
    if not _PRICE_SHAPE.match(text):
        return None

    normalized = _normalize_separators(text)
    if normalized is None:
        return None

    try:
        return PriceParse(value=Decimal(normalized), repaired=repaired)
    except InvalidOperation:
        return None


def _normalize_separators(text: str) -> str | None:
    """Resolve '.' and ',' into a single decimal point.

    The ambiguous case is a lone separator followed by digits:
      "1,299" -> thousands separator (3 digits follow)
      "12,99" -> European decimal comma (2 digits follow)
    Three trailing digits means grouping; one or two means a decimal part.
    """
    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        # Rightmost separator is the decimal point; the other groups.
        if text.rfind(".") > text.rfind(","):
            return text.replace(",", "")
        return text.replace(".", "").replace(",", ".")

    if has_comma:
        tail = text.split(",")[-1]
        if len(tail) == 3 and text.count(",") >= 1:
            return text.replace(",", "")  # grouping
        return text.replace(",", ".")  # decimal comma

    if has_dot:
        tail = text.split(".")[-1]
        if len(tail) == 3:
            # Ambiguous and dangerous. "1.299" could be European grouping
            # for 1299; "12.999" is a three-decimal value that is not
            # money. Nothing in the token distinguishes them, and guessing
            # grouping turns 12.999 into 12999 -- a thousand-fold error
            # that reconciliation would blame on the wrong field. Reject
            # and let the caller surface it for review.
            #
            # Genuine European grouping arrives with a decimal part too
            # ("1.299,50") and is handled by the both-separators branch above.
            return None
        return text

    return text


def _is_bare_integer(text: str) -> bool:
    return text.isdigit()


def find_price_column(
    rows: list[Row],
    *,
    tolerance_ratio: float = 0.04,
) -> float | None:
    """Find the x-coordinate of the right-aligned price column.

    Uses each candidate's *right* edge, not its left. Prices are
    right-aligned, so "9.50" and "112.00" share a right edge but start at
    different x-positions -- clustering on left edges would split them into
    two columns.

    Returns None when no column is evident. That happens on receipts with
    too few price-shaped tokens to be confident, and the honest response is
    to say so rather than name an arbitrary x.
    """
    if not rows:
        return None

    page_right = max(row.bbox.x1 for row in rows)
    page_left = min(row.bbox.x0 for row in rows)
    page_width = page_right - page_left
    if page_width <= 0:
        return None

    edges = [
        token.bbox.x1
        for row in rows
        for token in row.tokens
        if parse_price(token.text) is not None
    ]
    if len(edges) < 2:
        return None

    tolerance = page_width * tolerance_ratio
    cluster = _largest_cluster(sorted(edges), tolerance)

    # A single stray price does not establish a column.
    if len(cluster) < 2:
        return None

    return sum(cluster) / len(cluster)


def _largest_cluster(sorted_values: list[float], tolerance: float) -> list[float]:
    """Widest run of values within `tolerance` of the run's first element.

    Ties break toward the rightmost cluster: on a receipt the rightmost
    aligned column is the price column, and totals sitting slightly further
    right than line items should not pull the answer leftward.
    """
    best: list[float] = []
    current: list[float] = []

    for value in sorted_values:
        if current and value - current[0] > tolerance:
            if len(current) >= len(best):
                best = current
            current = []
        current.append(value)

    if len(current) >= len(best):
        best = current

    return best
