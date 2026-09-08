"""Group OCR tokens into visual rows.

OCR engines return tokens in loose reading order, but that order is not
reliable on receipts: a two-column layout (description on the left, price
on the right) often comes back interleaved or sorted purely by y-position
with ties broken arbitrarily.

So we ignore the order OCR gives us and reconstruct rows from geometry.
"""

from __future__ import annotations

from receipt_parser.models import BBox, Row, Token

# Two tokens are on the same line if their vertical spans overlap by at
# least this fraction of the shorter token's height. Expressed as a ratio
# rather than pixels so it holds regardless of image resolution or font size.
DEFAULT_OVERLAP_RATIO = 0.5


def cluster_rows(
    tokens: list[Token],
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[Row]:
    """Group tokens into rows, top to bottom, each ordered left to right.

    Greedy single pass over tokens sorted by vertical center. A token joins
    the open row if it overlaps that row vertically; otherwise it opens a
    new row.

    This handles the common receipt failure case where a description and
    its price are far apart horizontally but share a line -- pure
    reading-order grouping splits those into two rows and loses the
    association.
    """
    if not tokens:
        return []

    ordered = sorted(tokens, key=lambda t: (t.bbox.y_center, t.bbox.x0))

    rows: list[list[Token]] = [[ordered[0]]]
    open_bbox = ordered[0].bbox

    for token in ordered[1:]:
        if _same_line(open_bbox, token.bbox, overlap_ratio):
            rows[-1].append(token)
            open_bbox = _union(open_bbox, token.bbox)
        else:
            rows.append([token])
            open_bbox = token.bbox

    return [
        Row(tokens=sorted(group, key=lambda t: t.bbox.x0))
        for group in rows
    ]


def _same_line(a: BBox, b: BBox, overlap_ratio: float) -> bool:
    """Whether two boxes sit on the same visual line.

    Normalizes by the shorter height: a tall token (a large merchant name)
    should not swallow every short token that happens to fall within its
    vertical span.
    """
    shorter = min(a.height, b.height)
    if shorter <= 0:
        # Degenerate box from OCR. Containment has to be checked both ways:
        # a zero-height box spans a single point, so asking whether the
        # other box's center falls inside it is always False.
        return (a.y0 <= b.y_center <= a.y1) or (b.y0 <= a.y_center <= b.y1)

    return a.y_overlap(b) / shorter >= overlap_ratio


def _union(a: BBox, b: BBox) -> BBox:
    return BBox(
        x0=min(a.x0, b.x0),
        y0=min(a.y0, b.y0),
        x1=max(a.x1, b.x1),
        y1=max(a.y1, b.y1),
    )


def estimate_skew(tokens: list[Token]) -> float:
    """Estimate page rotation as pixels of vertical drift per 1000px of width.

    Operates on raw tokens, deliberately -- NOT on clustered rows.

    Skew is exactly what breaks row clustering: on a receipt photographed
    at an angle, two tokens on the same printed line drift apart
    vertically until their boxes no longer overlap, and cluster_rows()
    splits them. So measuring skew from clustered rows only works when
    there is no skew worth measuring.

    Instead: for each token, find tokens to its right inside a generous
    vertical band, and take the median slope across all such pairs. The
    band tolerates far more drift than clustering does, and the median
    discards the pairs that are genuinely on different lines.

    Callers should rotate the image and re-OCR when this exceeds roughly
    one median token height per 1000px, then cluster.
    """
    if len(tokens) < 2:
        return 0.0

    heights = sorted(t.bbox.height for t in tokens if t.bbox.height > 0)
    if not heights:
        return 0.0
    median_height = heights[len(heights) // 2]

    # Band is intentionally wide: we would rather admit some cross-line
    # pairs (the median removes them) than miss the real ones on a page
    # skewed badly enough to matter.
    band = median_height * 2.0
    min_span = median_height * 4.0  # ignore near-neighbours; slope is noise

    slopes: list[float] = []
    for i, a in enumerate(tokens):
        for b in tokens[i + 1 :]:
            dx = b.bbox.x0 - a.bbox.x0
            if abs(dx) < min_span:
                continue
            dy = b.bbox.y_center - a.bbox.y_center
            if abs(dy) > band:
                continue
            # Normalize direction so left-to-right slope is consistent.
            slopes.append((dy / dx) * 1000 if dx > 0 else (-dy / -dx) * 1000)

    if not slopes:
        return 0.0

    slopes.sort()
    return slopes[len(slopes) // 2]


def merge_price_fragments(rows: list[Row]) -> list[Row]:
    """Rejoin prices that OCR split into pieces.

    Tesseract regularly breaks an amount at its separator -- "$8,99"
    arrives as "$8" and ",99", "$3,49" as "$3," and "49". The pieces sit
    adjacent or even overlapping, and neither half parses as money on its
    own, so the row loses its price and drops out as noise.

    Worse, the split fragments carry low OCR confidence precisely because
    they are fragments. On a real Wendy's receipt the two halves of $3.49
    came back at 0.12 while every surrounding word scored above 0.95, so a
    confidence filter discards exactly the tokens that need repair. This
    runs before any confidence filtering, and the merged token inherits
    the highest confidence of its parts.

    Only merges when the join produces something that parses as a price
    and at least one half does not. Two legitimate adjacent numbers are
    left alone.
    """
    from receipt_parser.prices import parse_price

    merged_rows: list[Row] = []

    for row in rows:
        tokens = list(row.tokens)
        i = 0
        out: list[Token] = []

        while i < len(tokens):
            current = tokens[i]

            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                joined = current.text + nxt.text

                # Fragments sit tight against each other, and OCR
                # sometimes reports overlapping boxes for them, so a
                # negative gap is expected and allowed.
                gap = nxt.bbox.x0 - current.bbox.x1
                max_gap = max(current.bbox.height, nxt.bbox.height) * 0.6

                if (
                    gap < max_gap
                    and parse_price(joined) is not None
                    and parse_price(nxt.text) is None
                ):
                    out.append(
                        Token(
                            text=joined,
                            bbox=BBox(
                                x0=min(current.bbox.x0, nxt.bbox.x0),
                                y0=min(current.bbox.y0, nxt.bbox.y0),
                                x1=max(current.bbox.x1, nxt.bbox.x1),
                                y1=max(current.bbox.y1, nxt.bbox.y1),
                            ),
                            confidence=max(current.confidence, nxt.confidence),
                        )
                    )
                    i += 2
                    continue

            out.append(current)
            i += 1

        merged_rows.append(Row(tokens=out))

    return merged_rows
