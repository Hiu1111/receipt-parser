"""Parse a real receipt, then split it among people."""
import sys
from pathlib import Path

from receipt_parser.ocr.tesseract import TesseractProvider
from receipt_parser.pipeline import parse_image
from receipt_parser.split import Assignment, split_bill

receipt = parse_image(Path(sys.argv[1]).read_bytes(), TesseractProvider())

print(f"status: {receipt.status.value}")
for i, item in enumerate(receipt.line_items):
    print(f"  [{i}] {item.description:32} {item.total}")
print(f"subtotal={receipt.subtotal} tax={receipt.tax} "
      f"tip={receipt.tip} total={receipt.total}\n")

# Alternate items between two people; make the last one shared.
assignments = []
for i in range(len(receipt.line_items)):
    if i == len(receipt.line_items) - 1:
        assignments.append(Assignment(item_index=i, people=["alice", "bob"]))
    else:
        assignments.append(
            Assignment(item_index=i, people=["alice" if i % 2 == 0 else "bob"])
        )

result = split_bill(receipt, assignments)

for share in result.shares:
    print(f"  {share.name:8} items={share.items_subtotal:>8} "
          f"tax={share.tax:>6} tip={share.tip:>6} total={share.total:>8}")

print(f"\nshares sum to : {result.total}")
print(f"receipt total : {receipt.total}")
print("MATCH" if result.total == receipt.total else "MISMATCH")

for w in result.warnings:
    print("warn:", w)
