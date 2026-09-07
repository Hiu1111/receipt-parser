"""API tests.

The OCR provider is faked via dependency override. These tests are about
HTTP behaviour -- status codes, validation, error shapes -- and running
real OCR would make them slow and would couple CI to a system binary. The
end-to-end tests in test_pipeline.py cover the real engine.
"""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from receipt_parser.api import app, get_provider
from receipt_parser.models import BBox, Token

PRICE_X = 350.0


def tok(text: str, x0: float, y0: float, w: float = 40) -> Token:
    return Token(text=text, bbox=BBox(x0=x0, y0=y0, x1=x0 + w, y1=y0 + 12))


class FakeProvider:
    """Returns a fixed token layout regardless of input bytes."""

    def __init__(self, tokens=None):
        self.tokens = tokens if tokens is not None else _default_tokens()

    def extract(self, image_bytes: bytes) -> list[Token]:
        return self.tokens


def _default_tokens() -> list[Token]:
    def priced(text, price, y):
        width = len(price) * 10
        return [tok(text, 10, y), tok(price, PRICE_X - width, y, w=width)]

    rows = [
        [tok("JOES", 10, 40), tok("DINER", 60, 40)],
        priced("BURGER", "10.00", 100),
        priced("FRIES", "5.00", 130),
        priced("SUBTOTAL", "15.00", 160),
        priced("TAX", "1.20", 190),
        priced("TOTAL", "16.20", 220),
    ]
    return [t for row in rows for t in row]


@pytest.fixture
def client():
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(client, content=b"fake-image-bytes", content_type="image/png"):
    return client.post("/parse", files={"file": ("receipt.png", content, content_type)})


def test_health_reports_ocr_availability():
    """A container missing the Tesseract binary starts fine and then fails
    every request. The health check has to catch that.
    """
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "ocr_available" in body


def test_parse_returns_structured_receipt(client):
    response = upload(client)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["line_items"]) == 2
    assert Decimal(body["total"]) == Decimal("16.20")


def test_parse_rejects_empty_file(client):
    assert upload(client, content=b"").status_code == 400


def test_parse_rejects_wrong_content_type(client):
    assert upload(client, content_type="application/pdf").status_code == 415


def test_parse_rejects_oversized_upload(client):
    assert upload(client, content=b"x" * (16 * 1024 * 1024)).status_code == 413


def test_parse_requires_a_file(client):
    assert client.post("/parse").status_code == 422


def test_failed_parse_is_200_with_warnings():
    """A receipt that cannot be read is a result, not a server error."""
    app.dependency_overrides[get_provider] = lambda: FakeProvider(tokens=[])
    with TestClient(app) as client:
        response = upload(client)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["warnings"]


def test_provider_crash_returns_500_without_internals():
    """Parser internals in an error body help an attacker and not the user."""

    class Exploding:
        def extract(self, image_bytes):
            raise RuntimeError("tesseract segfaulted at 0xdeadbeef")

    app.dependency_overrides[get_provider] = lambda: Exploding()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = upload(client)
    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "deadbeef" not in response.text
    assert "segfault" not in response.text.lower()


def test_parse_then_split_round_trip(client):
    receipt = upload(client).json()
    response = client.post("/split", json={
        "receipt": receipt,
        "assignments": [
            {"item_index": 0, "people": ["alice"]},
            {"item_index": 1, "people": ["alice", "bob"]},
        ],
    })
    assert response.status_code == 200
    paid = sum(Decimal(s["total"]) for s in response.json()["shares"])
    assert paid == Decimal(receipt["total"])


def test_split_reports_unassigned_items(client):
    receipt = upload(client).json()
    body = client.post("/split", json={
        "receipt": receipt,
        "assignments": [{"item_index": 0, "people": ["alice"]}],
    }).json()
    assert body["unassigned_item_indices"] == [1]
    assert body["warnings"]


def test_split_rejects_assignment_with_no_people(client):
    receipt = upload(client).json()
    response = client.post("/split", json={
        "receipt": receipt,
        "assignments": [{"item_index": 0, "people": []}],
    })
    assert response.status_code == 422


def test_split_with_no_assignments(client):
    receipt = upload(client).json()
    body = client.post("/split", json={"receipt": receipt, "assignments": []}).json()
    assert body["shares"] == []
    assert body["warnings"]


def test_split_rejects_malformed_receipt(client):
    response = client.post("/split", json={
        "receipt": {"status": "ok", "total": None},
        "assignments": [],
    })
    assert response.status_code == 422
