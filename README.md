# receipt-parser

Structured line items from a photo of a receipt.

The hard part of splitting a bill is not the arithmetic — it's getting from
a crooked phone photo of faded thermal paper to a correct list of items and
prices. This is that layer.

## Pipeline

    photo → OCR → Token[] → layout → Row[] → prices → line items → ParsedReceipt

Each stage is a pure function over typed models, so every stage is testable
in isolation. No OCR credentials or network access are needed to run the
test suite.

## Status

| Stage | State |
|---|---|
| Data models | done |
| Row clustering from bounding boxes | done |
| Skew estimation | done |
| Price column detection | next |
| Line item extraction | planned |
| Arithmetic reconciliation | planned |
| OCR adapter | planned |
| HTTP API | planned |

## Design notes

**Row clustering ignores OCR reading order.** OCR engines return tokens in
loose reading order, but on a two-column receipt layout the item description
and its price are far apart horizontally, and the order they come back in is
not reliable. Rows are reconstructed from geometry instead: two tokens share
a line if their bounding boxes overlap vertically by at least half the
shorter token's height. Normalizing by the *shorter* height keeps a large
merchant header from swallowing every short line beneath it.

**Skew is measured on raw tokens, not clustered rows.** This is a real
ordering constraint, not a style choice. On a receipt photographed at an
angle, two tokens on the same printed line drift apart vertically until
their boxes stop overlapping — so clustering splits them, every row ends up
with one token, and a row-based skew estimate reports zero drift on exactly
the images that have the most. `estimate_skew` works from the raw token
cloud with a deliberately wide vertical band and takes the median slope,
which tolerates the drift that clustering cannot.

**A parse without a total is not a successful parse.** `ParsedReceipt`
carries a status, and a model validator makes `status=OK` with no total
impossible to construct. Unparseable receipts exist — faded thermal paper,
abbreviated item names, bad lighting — and the honest response is to mark a
result unreconciled and surface the uncertain fields for review, not to
return confident numbers that happen to be wrong.

## Running

Requires Python 3.10+.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest tests/ -v

## Tests

The layout suite builds synthetic tokens directly rather than testing
against real images. Constructing an exact adversarial layout — a
zero-height OCR box, a 1px gap between dense thermal lines, a price emitted
before its description — takes three lines and pins the specific behavior.
Real receipt fixtures come in at the accuracy-measurement stage, where they
measure something the synthetic cases can't.
