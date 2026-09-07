"""Tesseract adapter.

Free, offline, and adequate on clean printed receipts. Noticeably worse
than hosted engines on faded thermal paper, which is the case that
matters most -- so this is the development and test engine, with a hosted
provider swappable in behind the same interface for production.
"""

from __future__ import annotations

import io

from receipt_parser.models import BBox, Token
from receipt_parser.ocr.base import DEFAULT_MIN_CONFIDENCE


class TesseractProvider:
    """Wraps pytesseract's word-level output.

    `--psm 6` tells Tesseract to treat the image as a single uniform block
    of text. The default mode tries to detect columns and reading order,
    which on a receipt splits the description column and the price column
    into separate blocks and destroys the horizontal relationship the
    layout stage depends on.
    """

    def __init__(
        self,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        preprocess: bool = True,
        psm: int = 6,
    ) -> None:
        self.min_confidence = min_confidence
        self.preprocess = preprocess
        self.psm = psm

    def extract(self, image_bytes: bytes) -> list[Token]:
        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        if self.preprocess:
            image = _preprocess(image)

        data = pytesseract.image_to_data(
            image,
            config=f"--psm {self.psm}",
            output_type=pytesseract.Output.DICT,
        )

        tokens: list[Token] = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue

            # Tesseract reports -1 for non-text regions and 0-100 for words.
            raw_conf = float(data["conf"][i])
            if raw_conf < 0:
                continue
            confidence = raw_conf / 100.0
            if confidence < self.min_confidence:
                continue

            left, top = data["left"][i], data["top"][i]
            width, height = data["width"][i], data["height"][i]

            tokens.append(
                Token(
                    text=text,
                    bbox=BBox(
                        x0=float(left),
                        y0=float(top),
                        x1=float(left + width),
                        y1=float(top + height),
                    ),
                    confidence=confidence,
                )
            )

        return tokens


def _preprocess(image):
    """Grayscale and upscale.

    Thermal receipts are low-contrast and often photographed small.
    Tesseract's accuracy drops sharply below roughly 300 DPI equivalent,
    and upscaling a small image recovers a surprising amount -- it gives
    the character classifier more pixels to work with even though no new
    information was added.

    Binarization is deliberately not done here. A global threshold
    destroys faded text on unevenly lit photos, which is exactly the
    population this needs to handle.
    """
    from PIL import Image

    image = image.convert("L")

    min_width = 1000
    if image.width < min_width:
        scale = min_width / image.width
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return image
