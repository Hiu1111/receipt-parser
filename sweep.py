"""Try OCR settings against a receipt to see which recovers the most."""
import io
import sys
from pathlib import Path

import pytesseract
from PIL import Image

from receipt_parser.models import BBox, Token
from receipt_parser.pipeline import parse_tokens

path = Path(sys.argv[1])
raw = Image.open(io.BytesIO(path.read_bytes())).convert("L")


def run(psm: int, min_width: int):
    img = raw
    if img.width < min_width:
        scale = min_width / img.width
        img = img.resize((min_width, int(img.height * scale)), Image.LANCZOS)

    d = pytesseract.image_to_data(
        img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    tokens = []
    for i in range(len(d["text"])):
        text = d["text"][i].strip()
        conf = float(d["conf"][i])
        if not text or conf < 0:
            continue
        tokens.append(Token(
            text=text,
            bbox=BBox(x0=d["left"][i], y0=d["top"][i],
                      x1=d["left"][i] + d["width"][i],
                      y1=d["top"][i] + d["height"][i]),
            confidence=conf / 100.0,
        ))

    r = parse_tokens(tokens)
    print(f"psm={psm} width={min_width:5}  {r.status.value:13} "
          f"items={len(r.line_items)} sub={r.subtotal} total={r.total}")


for psm in (4, 6, 11, 12):
    for width in (1000, 2000, 3000):
        try:
            run(psm, width)
        except Exception as e:
            print(f"psm={psm} width={width}: {type(e).__name__}")
