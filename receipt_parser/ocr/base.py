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


# Deliberately permissive. The obvious setting is something like 0.30, to
# drop OCR hallucinating structure out of speckles and paper edges -- but
# split price fragments score *lower* than noise does. On a real Wendy's
# receipt the two halves of "$3,49" came back at 0.12 while every word
# around them scored above 0.95, and a 0.30 filter silently deleted a line
# item.
#
# So the floor is zero: keep everything Tesseract believes is text at all.
# (It reports -1 for non-text regions, which the provider rejects
# separately, so this is not the same as keeping literally everything.)
# Junk is rejected downstream instead, where there is more to judge on --
# whether the token parses as money, whether it sits in the price column,
# and whether the row's arithmetic reconciles. Position and structure are
# better evidence than the engine's own confidence score.
DEFAULT_MIN_CONFIDENCE = 0.0
