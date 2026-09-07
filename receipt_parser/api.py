"""HTTP interface.

Two endpoints and no database. Parsing is a pure function of the uploaded
image, and splitting is a pure function of a receipt plus assignments, so
neither needs server-side state. The client holds the parsed receipt
between the two calls.

That is a real tradeoff, not an omission. Statelessness means the service
scales horizontally with no coordination and has nothing to leak if it is
compromised -- and receipts are somebody's purchase history, which is
worth not storing. The cost is a larger request body on /split and no
history feature. If history is wanted later it belongs behind an
authenticated user account, which is a different design, not an extra
column.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from receipt_parser.models import ParsedReceipt
from receipt_parser.ocr.base import OCRProvider
from receipt_parser.ocr.tesseract import TesseractProvider
from receipt_parser.pipeline import parse_image
from receipt_parser.split import Assignment, SplitResult, split_bill

logger = logging.getLogger(__name__)

# Phone cameras produce 3-8MB images. Anything much larger is either a
# mistake or an attempt to exhaust memory in the OCR upscaling step.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/heic", "image/webp",
}

app = FastAPI(
    title="receipt-parser",
    description="Parse receipt images into structured line items and split them.",
    version="0.1.0",
)


def get_provider() -> OCRProvider:
    """Injected so tests can substitute a fake and CI needs no Tesseract."""
    return TesseractProvider()


class SplitRequest(BaseModel):
    receipt: ParsedReceipt
    assignments: list[Assignment] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    ocr_available: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Reports whether OCR is actually usable, not just whether the process is up.

    Tesseract is a separate binary. A container that starts fine but is
    missing it would pass a naive health check and then fail every real
    request.
    """
    import shutil

    return HealthResponse(
        status="ok",
        ocr_available=shutil.which("tesseract") is not None,
    )


@app.post("/parse", response_model=ParsedReceipt)
async def parse(
    provider: Annotated[OCRProvider, Depends(get_provider)],
    file: Annotated[UploadFile, File(description="Receipt photo")],
) -> ParsedReceipt:
    """Parse a receipt image into structured line items.

    Returns 200 even when the parse fails. A failed parse is a result, not
    a server error -- the response carries the status and the warnings
    explaining what went wrong, and the client needs those to show a
    review screen. Reserving non-2xx for genuine faults keeps that
    distinction usable.
    """
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    try:
        return parse_image(contents, provider)
    except Exception:
        # Log the detail, return a generic message. Parser internals in an
        # error body tell an attacker about the stack and tell a user
        # nothing they can act on.
        logger.exception("Parse failed for upload %s", file.filename)
        raise HTTPException(
            status_code=500, detail="Could not process the image."
        )


@app.post("/split", response_model=SplitResult)
def split(request: SplitRequest) -> SplitResult:
    """Divide a parsed receipt among people.

    Takes the receipt back rather than looking it up, because the service
    stores nothing. Assignments reference line items by index into the
    receipt the client already holds.
    """
    return split_bill(request.receipt, request.assignments)
