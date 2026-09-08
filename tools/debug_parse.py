"""Show what the parser sees, stage by stage.

Usage:
    python -m tools.debug_parse path/to/receipt.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from receipt_parser.items import classify_row, extract_items
from receipt_parser.layout import cluster_rows, merge_price_fragments
from receipt_parser.ocr.deskew import deskew
from receipt_parser.ocr.tesseract import TesseractProvider
from receipt_parser.prices import find_price_column, parse_price
from receipt_parser.reconcile import build_receipt, extract_totals


def main(path: str) -> None:
    image_bytes = Path(path).read_bytes()

    result = deskew(image_bytes)
    straightened, angle = result.image_bytes, result.angle
    print(f"deskew: {angle:+.2f} degrees  sharpness={result.sharpness:.2f}  warped={result.looks_warped}\n")

    tokens = TesseractProvider().extract(straightened)
    print(f"tokens: {len(tokens)}")

    rows = merge_price_fragments(cluster_rows(tokens))
    print(f"rows:   {len(rows)}")

    column = find_price_column(rows)
    print(f"price column x = {column}\n")

    if column is None:
        print("No price column found. Raw rows:")
        for i, row in enumerate(rows):
            print(f"  {i:3} {row.text}")
        return

    page_width = max(r.bbox.x1 for r in rows) - min(r.bbox.x0 for r in rows)
    tolerance = max(page_width * 0.04, 1.0)

    print(f"{'#':>3}  {'kind':<9} {'right-edge':>10}  text")
    print("-" * 72)
    for i, row in enumerate(rows):
        kind = classify_row(row, column, tolerance)
        last = row.tokens[-1]
        edge = last.bbox.x1
        in_col = "IN " if abs(edge - column) <= tolerance else "OUT"
        print(f"{i:>3}  {kind.value:<9} {edge:>7.0f} {in_col}  {row.text}")

    items, summary = extract_items(rows, column, page_width=page_width)
    totals = extract_totals(rows, summary, column, tolerance)

    print(f"\nsummary rows: {summary}")
    print(f"subtotal={totals.subtotal} tax={totals.tax} "
          f"tip={totals.tip} total={totals.total}")

    receipt = build_receipt(rows, items, summary, column, tolerance=tolerance)
    print(f"\nstatus: {receipt.status.value}")
    for item in receipt.line_items:
        print(f"  item: {item.description!r} = {item.total}")
    for warning in receipt.warnings:
        print(f"  warn: {warning}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
