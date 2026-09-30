import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import MAX_UPLOAD_BYTES, app

APARTMENTS = [
    {"apartment_id": "APT-00000", "image_path": "train/kitchen/a.jpg", "room_type": "kitchen",
     "price_mln": 10.0, "area_sqm": 50, "rooms": 2},
    {"apartment_id": "APT-00001", "image_path": "train/bedroom/b.jpg", "room_type": "bedroom",
     "price_mln": 20.0, "area_sqm": 70, "rooms": 3},
    {"apartment_id": "APT-00002", "image_path": "train/kitchen/c.jpg", "room_type": "kitchen",
     "price_mln": 30.0, "area_sqm": 90, "rooms": 3},
]


class FakeEngine:
    """Подменяет src.search.SearchEngine: тестируем HTTP-слой без модели и индекса."""
    size = len(APARTMENTS)

    def search(self, image, k, apartment_filter=None):
        candidates = [a for a in APARTMENTS if apartment_filter is None or apartment_filter(a)]
        hits = [SimpleNamespace(similarity=1.0 - 0.1 * i, apartment=apt) for i, apt in enumerate(candidates[:k])]
        return SimpleNamespace(room_type="kitchen", confidence=0.9), hits


@pytest.fixture
def client():
    app.state.engine = FakeEngine()
    with TestClient(app) as client:
        yield client
    app.state.engine = None


def image_file(fmt="JPEG", content_type="image/jpeg"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format=fmt)
    return {"file": ("room.jpg", buffer.getvalue(), content_type)}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index_size": 3}


def test_search_returns_prediction_and_results(client):
    response = client.post("/search", files=image_file(), params={"k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query_room"] == {"room_type": "kitchen", "confidence": 0.9}
    assert [r["apartment"]["apartment_id"] for r in body["results"]] == ["APT-00000", "APT-00001"]
    assert body["results"][0]["image_url"].endswith("/images/train/kitchen/a.jpg")


def test_search_accepts_png(client):
    response = client.post("/search", files=image_file("PNG", "image/png"))
    assert response.status_code == 200


def test_search_applies_filters(client):
    response = client.post("/search", files=image_file(),
                           params={"room_type": "kitchen", "min_price": 15})
    assert response.status_code == 200
    assert [r["apartment"]["apartment_id"] for r in response.json()["results"]] == ["APT-00002"]


def test_search_rejects_unknown_room_type(client):
    response = client.post("/search", files=image_file(), params={"room_type": "garage"})
    assert response.status_code == 422


@pytest.mark.parametrize("k", [0, 51])
def test_search_rejects_invalid_k(client, k):
    response = client.post("/search", files=image_file(), params={"k": k})
    assert response.status_code == 422


def test_search_rejects_wrong_content_type(client):
    response = client.post("/search", files={"file": ("doc.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_search_rejects_corrupted_image(client):
    response = client.post("/search", files={"file": ("room.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400


def test_search_rejects_too_large_file(client):
    payload = b"\xff" * (MAX_UPLOAD_BYTES + 1)
    response = client.post("/search", files={"file": ("room.jpg", payload, "image/jpeg")})
    assert response.status_code == 413
