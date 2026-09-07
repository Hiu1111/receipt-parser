"""The boundary between OCR engines and the parser.

Everything downstream of this module consumes `Token` objects and knows
nothing about which engine produced them. That matters for two reasons.

First, engines disagree. Tesseract, Google Vision and PaddleOCR return
different coordinate conventions, different confidence scales, and
different ideas of what a "word" is. Normalizing once, here, keeps that
divergence out of the parsing logic.

Second, engine choice is a deployment decision, not a design one. Local
Tesseract costs nothing and runs offline; a hosted API is more accurate on
faded thermal paper but needs credentials and a network call. Both should
be swappable without touching a line of parsing code.
"""

from __future__ import annotations

from typing import Protocol

from receipt_parser.models import Token


class OCRProvider(Protocol):
    """Anything that can turn image bytes into positioned words."""

    def extract(self, image_bytes: bytes) -> list[Token]:
        ...


# Words below this confidence are usually OCR hallucinating structure out
# of noise -- speckles on thermal paper, the edge of the receipt, a
# fingertip in frame. Keeping them adds junk rows that the layout stage
# then has to reason about.
DEFAULT_MIN_CONFIDENCE = 0.30
