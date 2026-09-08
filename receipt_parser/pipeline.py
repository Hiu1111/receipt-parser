"""Image bytes to ParsedReceipt.

The one function that runs every stage in order. Kept deliberately thin --
all the judgment lives in the stages, and this just wires them and handles
the case where an early stage produces nothing usable.
"""

from __future__ import annotations

from receipt_parser.items import extract_items
from receipt_parser.layout import cluster_rows, estimate_skew, merge_price_fragments
from receipt_parser.models import ParsedReceipt, ParseStatus, Token
from receipt_parser.ocr.base import OCRProvider
from receipt_parser.ocr.deskew import deskew
from receipt_parser.prices import find_price_column
from receipt_parser.reconcile import build_receipt

# Drift beyond roughly one line-height per 1000px starts breaking row
# clustering. Below that, clustering absorbs it fine.
SKEW_WARNING_THRESHOLD = 12.0


def parse_tokens(tokens: list[Token]) -> ParsedReceipt:
    """Run the parsing stages over already-extracted tokens.

    Separate from `parse_image` so the whole pipeline can be tested with
    synthetic tokens, no OCR engine required.
    """
    if not tokens:
        return ParsedReceipt(
            status=ParseStatus.FAILED,
            warnings=["No text was found in the image."],
        )

    rows = cluster_rows(tokens)
    # Before anything reads a price: OCR splits amounts at their separator
    # often enough that skipping this loses whole line items.
    rows = merge_price_fragments(rows)
    price_column = find_price_column(rows)

    if price_column is None:
        return ParsedReceipt(
            status=ParseStatus.FAILED,
            warnings=[
                "Could not identify a price column. The image may be "
                "cropped, rotated, or not a receipt."
            ],
        )

    page_width = max(r.bbox.x1 for r in rows) - min(r.bbox.x0 for r in rows)
    tolerance = max(page_width * 0.04, 1.0)

    items, summary = extract_items(rows, price_column, page_width=page_width)
    receipt = build_receipt(rows, items, summary, price_column, tolerance=tolerance)

    skew = estimate_skew(tokens)
    if abs(skew) > SKEW_WARNING_THRESHOLD:
        receipt.warnings.append(
            f"Image appears rotated ({skew:.0f}px drift per 1000px). "
            "Straightening it may improve accuracy."
        )

    return receipt


def parse_image(
    image_bytes: bytes,
    provider: OCRProvider,
    *,
    auto_deskew: bool = True,
) -> ParsedReceipt:
    """Full pipeline: straighten the image if needed, OCR it, then parse.

    Deskew runs *before* OCR, not after. Rotation is what breaks this
    parser hardest -- measured on synthetic receipts, 2 degrees of tilt
    takes it from a clean parse to extracting nothing, because tokens on
    the same printed line stop overlapping vertically and every row
    collapses to a single token. Two degrees is well inside what a
    handheld photo produces.

    Correcting first also means OCR runs once, on a straight image, rather
    than once to measure and again to read.
    """
    angle = 0.0
    warped = False
    if auto_deskew:
        try:
            result = deskew(image_bytes)
            image_bytes, angle, warped = (
                result.image_bytes, result.angle, result.looks_warped
            )
        except Exception:
            # A malformed or unsupported image should fail in the OCR
            # provider with a useful error, not here in preprocessing.
            angle = 0.0

    receipt = parse_tokens(provider.extract(image_bytes))

    if angle:
        receipt.warnings.append(
            f"Image was rotated {angle:.1f} degrees before reading."
        )

    if warped:
        # Actionable, and worth saying even on a successful parse. Rotation
        # is correctable; curvature is not, and the fix is in the user's
        # hands rather than the parser's.
        receipt.warnings.append(
            "The receipt looks curled or photographed at an angle. Text "
            "lines that curve cannot be straightened by rotating, so some "
            "rows may be misread. Lay it flat and photograph it straight on."
        )

    return receipt
