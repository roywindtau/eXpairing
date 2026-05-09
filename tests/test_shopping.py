"""
tests/test_shopping.py
----------------------
Integration tests for the shopping-list router.
Uses FastAPI TestClient with an in-memory SQLite DB so tests are
fully isolated from the real fridge2fork.db.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, User
from backend.db.database import get_db
from backend.main import app


# ── in-memory DB fixtures ─────────────────────────────────────────────────────
# StaticPool makes every connection use the same underlying SQLite database so
# data created in one session (user fixture) is visible to the TestClient.

@pytest.fixture()
def engine():
    _engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(_engine)
    yield _engine
    Base.metadata.drop_all(_engine)


@pytest.fixture()
def SessionLocal(engine):
    return sessionmaker(bind=engine)


@pytest.fixture()
def db_session(SessionLocal):
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(SessionLocal):
    def override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def user(db_session) -> User:
    u = User(name="Test", beta=0.5)
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    return u


# ── GET /shopping/{user_id} ───────────────────────────────────────────────────

class TestGetShoppingList:
    def test_empty_list_for_new_user(self, client, user):
        res = client.get(f"/shopping/{user.id}")
        assert res.status_code == 200
        assert res.json() == []

    def test_404_for_unknown_user(self, client):
        res = client.get("/shopping/999999")
        assert res.status_code == 404

    def test_returns_added_items(self, client, user):
        client.post(f"/shopping/{user.id}", json={"ingredients": ["eggs", "milk"]})
        res = client.get(f"/shopping/{user.id}")
        names = [i["ingredient"] for i in res.json()]
        assert "eggs" in names
        assert "milk" in names


# ── POST /shopping/{user_id} ─────────────────────────────────────────────────

class TestAddToShoppingList:
    def test_adds_items_and_returns_201(self, client, user):
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["flour", "sugar"]})
        assert res.status_code == 201
        body = res.json()
        assert len(body["added"]) == 2
        assert body["skipped"] == []

    def test_deduplicates_on_second_add(self, client, user):
        client.post(f"/shopping/{user.id}", json={"ingredients": ["eggs"]})
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["eggs", "butter"]})
        body = res.json()
        assert len(body["added"]) == 1
        assert body["added"][0]["ingredient"] == "butter"
        assert "eggs" in body["skipped"]

    def test_deduplication_is_case_insensitive(self, client, user):
        client.post(f"/shopping/{user.id}", json={"ingredients": ["Eggs"]})
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["eggs"]})
        assert res.json()["skipped"] == ["eggs"]

    def test_stores_source_recipe(self, client, user):
        res = client.post(f"/shopping/{user.id}", json={
            "ingredients": ["onion"],
            "recipe_id": 42,
            "recipe_name": "French Onion Soup",
        })
        item = res.json()["added"][0]
        assert item["source_recipe_id"] == 42
        assert item["source_recipe_name"] == "French Onion Soup"

    def test_empty_ingredient_strings_are_ignored(self, client, user):
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["", "  ", "salt"]})
        assert len(res.json()["added"]) == 1

    def test_404_for_unknown_user(self, client):
        res = client.post("/shopping/999999", json={"ingredients": ["salt"]})
        assert res.status_code == 404

    def test_ingredients_stored_lowercased(self, client, user):
        client.post(f"/shopping/{user.id}", json={"ingredients": ["GARLIC"]})
        items = client.get(f"/shopping/{user.id}").json()
        assert items[0]["ingredient"] == "garlic"


# ── PATCH /shopping/{user_id}/{item_id} ──────────────────────────────────────

class TestToggleShoppingItem:
    def _add_item(self, client, user_id, ingredient="eggs"):
        res = client.post(f"/shopping/{user_id}", json={"ingredients": [ingredient]})
        return res.json()["added"][0]

    def test_check_item(self, client, user):
        item = self._add_item(client, user.id)
        res = client.patch(f"/shopping/{user.id}/{item['id']}", json={"is_checked": True})
        assert res.status_code == 200
        assert res.json()["is_checked"] is True

    def test_uncheck_item(self, client, user):
        item = self._add_item(client, user.id)
        client.patch(f"/shopping/{user.id}/{item['id']}", json={"is_checked": True})
        res = client.patch(f"/shopping/{user.id}/{item['id']}", json={"is_checked": False})
        assert res.json()["is_checked"] is False

    def test_404_for_unknown_item(self, client, user):
        res = client.patch(f"/shopping/{user.id}/999999", json={"is_checked": True})
        assert res.status_code == 404

    def test_cannot_toggle_other_users_item(self, client, db_session, user):
        other = User(name="Other", beta=0.5)
        db_session.add(other)
        db_session.commit()
        db_session.refresh(other)
        item = self._add_item(client, user.id)
        res = client.patch(f"/shopping/{other.id}/{item['id']}", json={"is_checked": True})
        assert res.status_code == 404


# ── DELETE /shopping/{user_id}/{item_id} — remove one ────────────────────────

class TestRemoveShoppingItem:
    def test_removes_item(self, client, user):
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["pepper"]})
        item_id = res.json()["added"][0]["id"]
        del_res = client.delete(f"/shopping/{user.id}/{item_id}")
        assert del_res.status_code == 204
        items = client.get(f"/shopping/{user.id}").json()
        assert not any(i["id"] == item_id for i in items)

    def test_404_for_already_removed_item(self, client, user):
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["pepper"]})
        item_id = res.json()["added"][0]["id"]
        client.delete(f"/shopping/{user.id}/{item_id}")
        res2 = client.delete(f"/shopping/{user.id}/{item_id}")
        assert res2.status_code == 404


# ── DELETE /shopping/{user_id} — clear ───────────────────────────────────────

class TestClearShoppingList:
    def test_clears_only_checked_items_by_default(self, client, user):
        res = client.post(f"/shopping/{user.id}", json={"ingredients": ["a", "b", "c"]})
        items = res.json()["added"]
        client.patch(f"/shopping/{user.id}/{items[0]['id']}", json={"is_checked": True})
        client.patch(f"/shopping/{user.id}/{items[1]['id']}", json={"is_checked": True})

        del_res = client.delete(f"/shopping/{user.id}?checked_only=true")
        assert del_res.status_code == 204

        remaining = client.get(f"/shopping/{user.id}").json()
        assert len(remaining) == 1
        assert remaining[0]["ingredient"] == "c"

    def test_clears_all_when_checked_only_false(self, client, user):
        client.post(f"/shopping/{user.id}", json={"ingredients": ["a", "b"]})
        client.delete(f"/shopping/{user.id}?checked_only=false")
        assert client.get(f"/shopping/{user.id}").json() == []

    def test_404_for_unknown_user(self, client):
        res = client.delete("/shopping/999999?checked_only=false")
        assert res.status_code == 404

    def test_clear_on_empty_list_is_noop(self, client, user):
        res = client.delete(f"/shopping/{user.id}?checked_only=true")
        assert res.status_code == 204
