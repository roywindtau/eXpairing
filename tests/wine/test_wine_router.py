"""
test_wine_router.py
-------------------
HTTP-level tests for backend/routers/wine.py.

Current scope: "Suggest me a wine" = top-N popular wines, plus wine rating.
No personalization / pairing yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.database import get_db
from backend.db.models import Base, Wine, WineEvent, User
from backend.main import app


# ── fixture ──────────────────────────────────────────────────────────────

def _seed(db):
    db.add_all([
        # 3 wines with distinct popularity
        Wine(id=1, name="Estate Malbec", style="Red",
             grapes_csv="Malbec", harmonize_csv="Beef,Lamb,Grilled",
             avg_rating=4.8, n_ratings=500),
        Wine(id=2, name="Coastal Sauvignon", style="White",
             grapes_csv="Sauvignon Blanc", harmonize_csv="Fish,Seafood",
             avg_rating=4.2, n_ratings=200),
        Wine(id=3, name="Sparkling Bubbly", style="Sparkling",
             grapes_csv="Chardonnay", harmonize_csv="Appetizer",
             avg_rating=3.5, n_ratings=10),
        User(id=42, beta=0.5),
    ])
    db.commit()


@pytest.fixture
def client():
    """TestClient with an in-memory DB."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    _seed(db)
    db.close()

    def _override_get_db():
        d = Session()
        try:
            yield d
        finally:
            d.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── GET /wine/ranked ────────────────────────────────────────────────────

def test_ranked_returns_valid_shape(client):
    r = client.get("/wine/ranked", params={"top_n": 5})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert 0 < len(data) <= 5
    item = data[0]
    for key in ("wine_id", "wine_name", "avg_rating", "n_ratings",
                "style", "variety", "harmonize_csv"):
        assert key in item, f"missing field '{key}'"


def test_ranked_ordered_by_popularity(client):
    """Higher Bayesian-smoothed popularity ranks first."""
    r = client.get("/wine/ranked", params={"top_n": 3})
    data = r.json()
    # Estate Malbec (4.8, n=500) should beat the others.
    assert data[0]["wine_id"] == 1


def test_ranked_top_n_truncation(client):
    r = client.get("/wine/ranked", params={"top_n": 2})
    assert len(r.json()) == 2


def test_ranked_default_top_n(client):
    r = client.get("/wine/ranked")
    assert r.status_code == 200
    assert len(r.json()) == 3   # only 3 seeded, default cap is 10


# ── POST /wine-events ───────────────────────────────────────────────────

def test_post_wine_event_creates_row(client):
    r = client.post("/wine-events", json={
        "user_id": 42, "wine_id": 1, "event_type": "rate", "rating": 4.5,
    })
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "ok"
    assert "event_id" in body


def test_post_wine_event_unknown_wine_returns_404(client):
    r = client.post("/wine-events", json={
        "user_id": 42, "wine_id": 999999, "event_type": "rate", "rating": 4.0,
    })
    assert r.status_code == 404


def test_post_wine_event_rejects_invalid_event_type(client):
    r = client.post("/wine-events", json={
        "user_id": 42, "wine_id": 1, "event_type": "cook", "rating": 4.0,
    })
    assert r.status_code == 422


def test_post_wine_event_rejects_missing_rating(client):
    r = client.post("/wine-events", json={
        "user_id": 42, "wine_id": 1, "event_type": "rate",
    })
    assert r.status_code == 422


def test_post_wine_event_rejects_out_of_range_rating(client):
    r = client.post("/wine-events", json={
        "user_id": 42, "wine_id": 1, "event_type": "rate", "rating": 6.5,
    })
    assert r.status_code == 422


def test_post_wine_event_writes_explicit_not_synthetic(client):
    r = client.post("/wine-events", json={
        "user_id": 42, "wine_id": 1, "event_type": "rate", "rating": 4.0,
    })
    assert r.status_code == 201

    db_gen = app.dependency_overrides[get_db]()
    db = next(db_gen)
    try:
        ev = (
            db.query(WineEvent)
            .filter(WineEvent.user_id == 42, WineEvent.wine_id == 1)
            .first()
        )
        assert ev is not None
        assert ev.synthetic is False
        assert ev.rating == 4.0
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
