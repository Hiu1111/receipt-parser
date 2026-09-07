"""Data models for receipt parsing.

The pipeline is: OCR -> Token[] -> Row[] -> ParsedReceipt

Every stage is a pure function over these types, which means each one is
testable in isolation without OCR credentials or a live API.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    """Axis-aligned bounding box in image pixel coordinates.

    Origin is top-left, y increases downward (the convention every OCR
    engine uses).
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    def y_overlap(self, other: BBox) -> float:
        """Vertical overlap with another box, in pixels. 0 if disjoint."""
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))


class Token(BaseModel):
    """A single word from OCR, with its position on the page."""

    text: str
    bbox: BBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Row(BaseModel):
    """Tokens that sit on the same visual line, ordered left to right."""

    tokens: list[Token]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def bbox(self) -> BBox:
        return BBox(
            x0=min(t.bbox.x0 for t in self.tokens),
            y0=min(t.bbox.y0 for t in self.tokens),
            x1=max(t.bbox.x1 for t in self.tokens),
            y1=max(t.bbox.y1 for t in self.tokens),
        )

    @property
    def min_confidence(self) -> float:
        return min(t.confidence for t in self.tokens)


class LineItem(BaseModel):
    """One purchased item."""

    description: str
    quantity: int = 1
    unit_price: Decimal | None = None
    total: Decimal
    # Which source rows produced this item. Kept so the review UI can
    # highlight the original pixels when a user corrects a field.
    source_row_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ParseStatus(str, Enum):
    OK = "ok"
    # Parsed, but the arithmetic doesn't reconcile. Never present these
    # numbers as authoritative.
    UNRECONCILED = "unreconciled"
    # Could not extract a usable structure at all.
    FAILED = "failed"


class ParsedReceipt(BaseModel):
    merchant: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    tip: Decimal | None = None
    total: Decimal | None = None

    status: ParseStatus = ParseStatus.OK
    # Human-readable reasons the parse is suspect. Drives the review UI.
    warnings: list[str] = Field(default_factory=list)

    @property
    def items_sum(self) -> Decimal:
        return sum((i.total for i in self.line_items), Decimal("0"))

    @model_validator(mode="after")
    def _never_ok_without_total(self) -> ParsedReceipt:
        """A receipt with no total is not a successful parse.

        Enforced at the type level so no code path can construct an
        over-confident result by accident.
        """
        if self.status is ParseStatus.OK and self.total is None:
            raise ValueError("status=OK requires a total")
        return self
