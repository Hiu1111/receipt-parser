"""End-to-end tests: rendered image -> OCR -> parsed receipt.

These are the only tests that need Tesseract installed. Everything else in
the suite runs on synthetic tokens and stays fast and dependency-free;
these exist to prove the wiring is real and to pin the degradation limits
so a regression in preprocessing shows up as a failing test rather than
as worse accuracy nobody noticed.
"""

from __future__ import annotations

import io
import shutil
from decimal import Decimal

import pytest
from PIL import Image

from receipt_parser.models import ParseStatus
from receipt_parser.ocr.deskew import find_skew_angle
from receipt_parser.pipeline import parse_image, parse_tokens
from tests.receipt_render import ReceiptSpec, render

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="Tesseract not installed",
)


@pytest.fixture(scope="module")
def provider():
    from receipt_parser.ocr.tesseract import TesseractProvider

    return TesseractProvider()


@pytest.fixture
def spec():
    return ReceiptSpec(
        items=[("BURGER", "12.99"), ("FRIES", "4.50"), ("SODA", "3.25")],
        tax="1.66",
        tip="4.00",
    )


def assert_matches(receipt, spec: ReceiptSpec):
    assert receipt.status is ParseStatus.OK, receipt.warnings
    assert len(receipt.line_items) == len(spec.items)
    assert receipt.total == spec.total
    assert receipt.subtotal == spec.subtotal


def test_clean_receipt_end_to_end(provider, spec):
    receipt = parse_image(render(spec), provider)
    assert_matches(receipt, spec)
    assert receipt.merchant is not None
    assert "DINER" in receipt.merchant.upper()


def test_line_item_values(provider, spec):
    receipt = parse_image(render(spec), provider)
    by_price = {i.total for i in receipt.line_items}
    assert by_price == {Decimal("12.99"), Decimal("4.50"), Decimal("3.25")}


def test_receipt_without_subtotal_line(provider):
    """Fast-food receipts frequently omit the subtotal."""
    spec = ReceiptSpec(
        items=[("COFFEE", "3.50"), ("BAGEL", "2.75")],
        tax="0.44",
        include_subtotal=False,
    )
    receipt = parse_image(render(spec), provider)
    assert receipt.status is ParseStatus.OK
    assert receipt.total == spec.total


@pytest.mark.parametrize("rotation", [0, 1, 2, 3, 5, 8, 12, -3, -7])
def test_skew_angle_is_measured_accurately(spec, rotation):
    """Projection-profile detection recovers the angle to within 0.5 degrees.

    The correction is the negative of the applied rotation.
    """
    image = Image.open(io.BytesIO(render(spec, rotate=rotation)))
    detected = find_skew_angle(image)
    assert abs(detected - (-rotation)) < 0.5


@pytest.mark.parametrize("rotation", [1, 2, 3, 5, 8, 12, -5])
def test_deskew_rescues_rotated_receipts(provider, spec, rotation):
    """Without correction, 2 degrees is enough to extract nothing at all.

    Tokens on the same printed line drift apart vertically until their
    boxes stop overlapping, so row clustering splits every line into
    single-token rows and no description ever meets its price.
    """
    receipt = parse_image(render(spec, rotate=rotation), provider)
    assert_matches(receipt, spec)


def test_rotation_without_deskew_fails(provider, spec):
    """Pins the problem deskew solves. If this ever passes, the parser
    itself became rotation-tolerant and the preprocessing may be redundant.
    """
    receipt = parse_image(render(spec, rotate=5), provider, auto_deskew=False)
    assert receipt.status is not ParseStatus.OK


def test_deskew_is_reported_in_warnings(provider, spec):
    receipt = parse_image(render(spec, rotate=6), provider)
    assert any("rotated" in w.lower() for w in receipt.warnings)


def test_straight_image_is_not_rotated(provider, spec):
    receipt = parse_image(render(spec), provider)
    assert not any("rotated" in w.lower() for w in receipt.warnings)


@pytest.mark.parametrize("blur", [0.5, 1.0, 1.5])
def test_tolerates_moderate_blur(provider, spec, blur):
    receipt = parse_image(render(spec, blur=blur), provider)
    assert receipt.total == spec.total


@pytest.mark.parametrize("contrast", [0.6, 0.4, 0.25, 0.15])
def test_tolerates_faded_thermal_paper(provider, spec, contrast):
    """Contrast below 1 fades ink toward the paper, as thermal receipts do."""
    receipt = parse_image(render(spec, contrast=contrast), provider)
    assert_matches(receipt, spec)


def test_heavy_blur_fails_loudly(provider, spec):
    """Unreadable input must not produce a confident wrong answer."""
    receipt = parse_image(render(spec, blur=4.0), provider)
    assert receipt.status is not ParseStatus.OK
    assert receipt.warnings


def test_blank_image_fails(provider):
    blank = Image.new("L", (600, 800), color=255)
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")

    receipt = parse_image(buffer.getvalue(), provider)
    assert receipt.status is ParseStatus.FAILED
    assert receipt.warnings


def test_no_tokens_fails_without_crashing():
    receipt = parse_tokens([])
    assert receipt.status is ParseStatus.FAILED
