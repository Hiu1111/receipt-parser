# receipt-parser

Photograph a receipt, say who had what, get what each person owes.

The arithmetic is the easy part. The hard part is getting from a crooked
phone photo of faded thermal paper to a correct list of items and prices,
and knowing when you have failed.

![The split screen](docs/screenshot.png)

## Pipeline

Every stage after OCR is a pure function over typed models, so the whole
parser is testable without an OCR engine, credentials, or network access.
149 of the 157 tests run that way; the other 8 need Tesseract and are
skipped automatically when it is missing.

## Running

Requires Python 3.10+ and Tesseract (`brew install tesseract`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn receipt_parser.api:app --reload
```

Open http://localhost:8000. Or with Docker:

```bash
docker build -t receipt-parser .
docker run -p 8000:8000 receipt-parser
```

The API is at `/docs`. Two endpoints: `POST /parse` takes an image and
returns structured line items; `POST /split` takes a receipt plus item
assignments and returns what each person owes.

## Results on real receipts

| Receipt | Condition | Status | Items | Total |
|---|---|---|---|---|
| Retail template | Printed, clean | ok | 7/7 | correct |
| Walmart grocery | Thermal, photographed | ok | 13/13 | correct |
| Wendy's | Thermal, clean | ok | 3/3 | correct |
| Steakhouse | Curled, shot at an angle | failed | 5/17 | refused |
| Swiss cafe | German | failed | — | refused |

Three of five parse completely. Both failures are reported rather than
guessed at: the curled receipt returns a warning telling the user to lay
it flat, and the German one finds no total because the summary keywords
are English.

Every one of those receipts found at least one bug that synthetic test
images could not, which is the argument for testing against real input
early.

## Design notes

**Skew correction measures pixels, not OCR tokens.** The first
implementation estimated rotation from token positions: find pairs of
words on the same printed line, take the median slope. It tracked true
rotation to within 4% at 3 degrees, then broke -- at 5 degrees it
reported 16px of drift where the truth was 88px, and at 8 degrees it
reported almost none.

The cause is that token-based estimation pairs words within a vertical
band. As rotation increases, words genuinely on the same line drift out
of the band while words on adjacent lines drift into it, so the median
fills with cross-line pairs and collapses toward zero. It fails by
reporting a small skew rather than an unknown one, which is worse.

Replacing it with a projection-profile search over the image pixels --
rotate through candidate angles, keep the one where the horizontal ink
profile has the highest variance -- moved the working range from 2
degrees to beyond 12. Two degrees of tilt previously took the parser from
a clean parse to extracting nothing at all, because tokens on the same
line stop overlapping vertically and every row collapses to a single
token. Two degrees is well inside what a handheld photo produces.

**The OCR confidence floor is zero, on purpose.** The obvious setting is
something like 0.3, to drop the tokens OCR hallucinates out of speckles
and paper edges. That setting silently deleted a line item.

Tesseract splits amounts at their separator -- "$3,49" comes back as
"$3," and "49" -- and the fragments score *low confidence precisely
because they are fragments*. On a real Wendy's receipt the two halves of
$3.49 scored 0.12 while every word around them scored above 0.95. A
confidence filter discards exactly the tokens that need repair.

So fragments are merged first, by checking whether two adjacent tokens
join into something that parses as money when neither half does, and
junk is rejected downstream instead -- on whether a token parses as a
price, whether it sits in the price column, and whether the row's
arithmetic reconciles. Position and structure turned out to be better
evidence than the engine's own confidence score.

**The receipt is its own checksum.** Row clustering, price parsing and
item extraction are all heuristics, and each fails quietly: a missed
item, a misread digit, a description absorbed into the wrong row. None of
those announce themselves, and every individual value still looks
plausible.

Arithmetic does announce them. Items sum to the subtotal, and subtotal
plus tax plus tip equals the total. When those do not hold, something
upstream is wrong. A `ParsedReceipt` carries a status, and a model
validator makes `status=OK` with no total impossible to construct -- so
no code path can return a confident answer that was never checked. On the
steakhouse receipt this is what caught the failure: the warning names the
missing amount to the cent.

**Ambiguity is refused, not guessed.** A lone dot followed by three
digits is genuinely undecidable -- "1.299" could be European grouping for
1299, "12.999" is a three-decimal value that is not money, and nothing in
the token distinguishes them. Guessing grouping turns 12.999 into 12999,
a thousand-fold error that reconciliation would then blame on the wrong
field. The parser returns nothing for these and lets the caller surface
them for review.

**Shares always sum to the total.** Tax and tip are allocated in
proportion to what each person ordered rather than split evenly, so the
person who had a salad does not subsidise the person who had a steak.
Both allocation and even splits round, and rounded parts do not have to
sum back. A $10 item split three ways becomes 3.34, 3.33, 3.33 -- not
three 3.33s that lose a penny. Largest-remainder allocation guarantees
the shares reconcile to the receipt total exactly, which is the one bug
a user would notice immediately.

## Known limitations

**Curled or angled photos.** Rotation is correctable; curvature is not.
When the projection profile has no clear peak at any angle, the parser
detects it and tells the user to lay the receipt flat, but it does not
dewarp. Full perspective and curve correction is a separate project.

**English only.** Summary keywords (`SUBTOTAL`, `TAX`, `TOTAL`) are
English. A German receipt parses items but finds no totals block and
correctly returns `failed`.

**Wrapped descriptions.** An item whose name spans two lines is read from
the second line only. The price and total are correct; the description
is truncated.

**Character-level OCR errors.** "GV PARM 160Z" reads as "160Z" or "1602"
depending on the engine. Prices are validated by reconciliation;
descriptions are not, so they carry whatever OCR produced.

**Tip allocation is untested against a real tip line.** The logic has
unit tests and works on synthetic receipts, but none of the five real
receipts had a tip.

## Layout

`python -m tools.debug_parse receipt.jpg` is the fastest way to see where
a bad parse went wrong -- it prints the deskew angle, every clustered
row, how each row was classified, and whether its amount landed in the
price column.
