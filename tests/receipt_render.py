"""Render receipt images from known data.

Real receipts have no ground truth attached. You can photograph fifty of
them and still not have a machine-checkable answer for what the parser
should produce -- someone has to type it all in by hand.

Rendering solves that: the data that produced the image *is* the expected
output. That makes an end-to-end accuracy test possible today, without a
corpus, and it makes degradation testable -- blur the same image, rotate
it, drop the contrast, and measure exactly where the parser stops coping.

What this cannot do is substitute for real receipts. A rendered image has
even lighting, a clean font, and no creases. Passing here means the
wiring is correct, not that the parser handles a crumpled thermal receipt
from a diner.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from decimal import Decimal

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


@dataclass
class ReceiptSpec:
    """Ground truth for a rendered receipt."""

    merchant: str = "JOE'S DINER"
    address: str = "142 MAIN ST"
    phone: str = "860-555-0142"
    items: list[tuple[str, str]] = field(default_factory=list)
    tax: str | None = None
    tip: str | None = None
    include_subtotal: bool = True
    footer: str = "THANK YOU"

    @property
    def subtotal(self) -> Decimal:
        return sum((Decimal(p) for _, p in self.items), Decimal("0"))

    @property
    def total(self) -> Decimal:
        out = self.subtotal
        if self.tax:
            out += Decimal(self.tax)
        if self.tip:
            out += Decimal(self.tip)
        return out


def _load_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(
    spec: ReceiptSpec,
    *,
    width: int = 640,
    font_size: int = 22,
    rotate: float = 0.0,
    blur: float = 0.0,
    contrast: float = 1.0,
) -> bytes:
    """Render a receipt to PNG bytes.

    The degradation parameters exist to find the parser's breaking point:
    `rotate` for a photo taken at an angle, `blur` for a bad focus,
    `contrast` for faded thermal paper.
    """
    font = _load_font(font_size)
    line_height = int(font_size * 1.6)
    margin = 40

    lines: list[tuple[str, str | None]] = [
        (spec.merchant, None),
        (spec.address, None),
        (spec.phone, None),
        ("", None),
    ]
    lines += [(desc, price) for desc, price in spec.items]
    lines.append(("", None))

    if spec.include_subtotal:
        lines.append(("SUBTOTAL", str(spec.subtotal)))
    if spec.tax:
        lines.append(("TAX", spec.tax))
    if spec.tip:
        lines.append(("TIP", spec.tip))
    lines.append(("TOTAL", str(spec.total)))
    lines.append(("", None))
    lines.append((spec.footer, None))

    height = margin * 2 + line_height * len(lines)
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    y = margin
    for text, price in lines:
        if text:
            draw.text((margin, y), text, font=font, fill=0)
        if price is not None:
            price_width = draw.textlength(price, font=font)
            draw.text((width - margin - price_width, y), price, font=font, fill=0)
        y += line_height

    if rotate:
        image = image.rotate(rotate, expand=True, fillcolor=255, resample=Image.BICUBIC)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if contrast != 1.0:
        # Values below 1 fade the ink toward the paper, which is what
        # thermal receipts do as they age.
        image = image.point(lambda v: int(255 - (255 - v) * contrast))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
