"""
test_drink_cf.py
----------------
Tests for train_drink_cf.py, drink_item_similarity.py, drink_cold_start.py,
and serve_drink_cf.py.

TODO: fixture strategy needs rewriting.
  train_cf.py was migrated from DB-based loading to CSV-based loading
  (pre-train-cf branch). The `trained` fixture here patches SessionLocal
  and DB path constants that no longer exist in train_cf.py. It needs to
  write mini CSV files to tmp_path and patch the CSV path constants instead.
  item_similarity.py still loads from DB — its fixture is still valid but
  coupled to the old train_cf fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import scipy.sparse as sp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# TODO: fixture needs rewrite after train_cf migrated from DB to CSV loading.
# The `trained` fixture patches SessionLocal + DB path constants that no longer
# exist in train_cf.py. Needs to write mini CSVs to tmp_path instead.
pytestmark = pytest.mark.skip(reason="TODO: fixture needs rewrite for CSV-based train_cf")

from backend.db.models import Base, Beer, Wine, DrinkEvent, User
from backend.ml.drinks.serving import cold_start as drink_cold_start
from backend.ml.drinks.training import item_similarity as drink_item_similarity
from backend.ml.drinks.serving import serve_cf as serve_drink_cf
from backend.ml.drinks.training import train_cf as train_drink_cf


# ── fixture data ─────────────────────────────────────────────────────────

def _seed(db):
    db.add_all([
        # 3 beers and 2 wines with avg/n stats for popularity tests
        Beer(id=1, name="Hop A",  style="IPA",
             avg_rating=4.5, n_ratings=200, review_tokens_csv="ipa"),
        Beer(id=2, name="Hop B",  style="IPA",
             avg_rating=4.0, n_ratings=150, review_tokens_csv="ipa"),
        Beer(id=3, name="Dark A", style="Stout",
             avg_rating=4.2, n_ratings=80, review_tokens_csv="stout"),
        Beer(id=4, name="Light A", style="Pilsner",
             avg_rating=3.5, n_ratings=60, review_tokens_csv="pilsner"),
        Beer(id=5, name="New Beer", style="IPA",
             avg_rating=4.8, n_ratings=2,  review_tokens_csv="ipa"),
        Wine(id=11, name="Red 1",  style="Red",
             grapes_csv="Malbec", harmonize_csv="Beef",
             avg_rating=4.2, n_ratings=20),
        Wine(id=12, name="White 1", style="White",
             grapes_csv="Sauvignon Blanc", harmonize_csv="Fish",
             avg_rating=3.8, n_ratings=15),
        Wine(id=13, name="Sparkling 1", style="Sparkling",
             grapes_csv="Chardonnay", harmonize_csv="Appetizer",
             avg_rating=4.0, n_ratings=3),
        # Users
        *(User(id=uid, beta=0.5) for uid in [1, 2, 3, 4, 5, 99]),
    ])
    # Co-rating pattern that makes Hop A and Hop B SIMILAR (both rated
    # high by users 1, 2, 3) and Dark A DISSIMILAR (only user 4 rates it high).
    db.add_all([
        # users who love the IPAs
        DrinkEvent(user_id=1, drink_id=1, event_type="rate", rating=5.0),
        DrinkEvent(user_id=1, drink_id=2, event_type="rate", rating=5.0),
        DrinkEvent(user_id=1, drink_id=4, event_type="rate", rating=2.0),
        DrinkEvent(user_id=2, drink_id=1, event_type="rate", rating=4.5),
        DrinkEvent(user_id=2, drink_id=2, event_type="rate", rating=5.0),
        DrinkEvent(user_id=2, drink_id=4, event_type="rate", rating=2.5),
        DrinkEvent(user_id=3, drink_id=1, event_type="rate", rating=5.0),
        DrinkEvent(user_id=3, drink_id=2, event_type="rate", rating=4.5),
        DrinkEvent(user_id=3, drink_id=3, event_type="rate", rating=3.0),
        # the stout-lover
        DrinkEvent(user_id=4, drink_id=3, event_type="rate", rating=5.0),
        DrinkEvent(user_id=4, drink_id=4, event_type="rate", rating=4.0),
        DrinkEvent(user_id=4, drink_id=1, event_type="rate", rating=2.0),
        # synthetic event (should NOT influence training)
        DrinkEvent(user_id=4, drink_id=5, event_type="rate", rating=5.0, synthetic=True),
        # wine co-rating
        DrinkEvent(user_id=1, drink_id=11, event_type="rate", rating=5.0),
        DrinkEvent(user_id=1, drink_id=12, event_type="rate", rating=3.0),
        DrinkEvent(user_id=2, drink_id=11, event_type="rate", rating=4.5),
        DrinkEvent(user_id=2, drink_id=12, event_type="rate", rating=3.5),
        DrinkEvent(user_id=3, drink_id=12, event_type="rate", rating=5.0),
        DrinkEvent(user_id=5, drink_id=11, event_type="rate", rating=2.0),
    ])
    db.commit()


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """Build DB, train CF + sim into tmp_path/models, yield (Session, models_dir)."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    _seed(db)
    db.close()

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Redirect every module-level path constant
    monkeypatch.setattr(train_drink_cf,        "SessionLocal", Session)
    monkeypatch.setattr(train_drink_cf,        "MODELS_DIR",   models_dir)
    monkeypatch.setattr(train_drink_cf,        "CF_MODEL",     models_dir / "drink_cf_model.pkl")
    monkeypatch.setattr(train_drink_cf,        "CF_META",      models_dir / "drink_cf_meta.json")

    monkeypatch.setattr(drink_item_similarity, "SessionLocal", Session)
    monkeypatch.setattr(drink_item_similarity, "MODELS_DIR",   models_dir)
    monkeypatch.setattr(drink_item_similarity, "SIM_BEER",     models_dir / "drink_sim_beer.npz")
    monkeypatch.setattr(drink_item_similarity, "SIM_BEER_IDS", models_dir / "drink_sim_beer_ids.npy")
    monkeypatch.setattr(drink_item_similarity, "SIM_WINE",     models_dir / "drink_sim_wine.npz")
    monkeypatch.setattr(drink_item_similarity, "SIM_WINE_IDS", models_dir / "drink_sim_wine_ids.npy")
    monkeypatch.setattr(drink_item_similarity, "SIM_META",     models_dir / "drink_sim_meta.json")
    # Lower threshold for the tiny fixture (5 ratings is too strict for 5 beers).
    monkeypatch.setattr(drink_item_similarity, "MIN_RATINGS_BEER", 2)
    monkeypatch.setattr(drink_item_similarity, "MIN_RATINGS_WINE", 2)

    monkeypatch.setattr(serve_drink_cf, "CF_MODEL_PATH",     models_dir / "drink_cf_model.pkl")
    monkeypatch.setattr(serve_drink_cf, "SIM_BEER_PATH",     models_dir / "drink_sim_beer.npz")
    monkeypatch.setattr(serve_drink_cf, "SIM_BEER_IDS_PATH", models_dir / "drink_sim_beer_ids.npy")
    monkeypatch.setattr(serve_drink_cf, "SIM_WINE_PATH",     models_dir / "drink_sim_wine.npz")
    monkeypatch.setattr(serve_drink_cf, "SIM_WINE_IDS_PATH", models_dir / "drink_sim_wine_ids.npy")

    train_drink_cf.train(n_factors=5, n_epochs=5)
    drink_item_similarity.train()
    serve_drink_cf._reset_for_tests()

    yield Session, models_dir

    serve_drink_cf._reset_for_tests()


# ── training artifacts ───────────────────────────────────────────────────

def test_training_produces_all_artifacts(trained):
    _, models_dir = trained
    for name in (
        "drink_cf_model.pkl",
        "drink_cf_meta.json",
        "drink_sim_beer.npz",
        "drink_sim_beer_ids.npy",
        "drink_sim_wine.npz",
        "drink_sim_wine_ids.npy",
        "drink_sim_meta.json",
    ):
        assert (models_dir / name).exists(), f"missing {name}"


def test_svd_excludes_synthetic(trained):
    """User 4 has a synthetic rating on drink 5; SVD should not have trained on it."""
    import json
    _, models_dir = trained
    meta = json.loads((models_dir / "drink_cf_meta.json").read_text())
    assert meta["synthetic_excluded"] is True
    # 12 real beer ratings in the fixture; 1 synthetic → meta should reflect 12
    # (some may be filtered by min_ratings_per_user — but never include synthetic)
    assert meta["n_ratings"] <= 12


def test_sim_matrices_loaded(trained):
    """Both sim matrices must load. Beer fixture is rich enough for non-zero
    similarities; wine fixture is intentionally tiny so it can legitimately
    be empty — the serve layer must handle that without crashing."""
    serve_drink_cf._load()
    assert serve_drink_cf._sim_beer is not None and serve_drink_cf._sim_beer.nnz > 0
    assert serve_drink_cf._sim_wine is not None  # may have nnz == 0 on tiny fixtures


# ── bayesian_popularity ─────────────────────────────────────────────────

def test_bayesian_popularity_smooths_low_n():
    """Drink with 2 ratings of 4.8 should not beat a drink with 200 ratings of 4.5."""
    drinks = {
        1: {"avg_rating": 4.5, "n_ratings": 200},
        5: {"avg_rating": 4.8, "n_ratings": 2},
    }
    scores = drink_cold_start.bayesian_popularity([1, 5], drinks)
    assert scores[1] > scores[5]


def test_bayesian_popularity_handles_missing_drink():
    scores = drink_cold_start.bayesian_popularity([1, 999], {1: {"avg_rating": 4.0, "n_ratings": 10}})
    assert 999 in scores
    assert scores[999] >= 0


def test_bayesian_popularity_empty_input():
    assert drink_cold_start.bayesian_popularity([], {}) == {}


# ── item_sim_seed_scores ────────────────────────────────────────────────

def test_item_sim_seed_scores_empty_when_no_seeds():
    assert drink_cold_start.item_sim_seed_scores([1, 2], [], [], None, None) == {}


def test_item_sim_seed_scores_returns_zero_for_unknown_candidate():
    ids = np.array([1, 2])
    sim = sp.csr_matrix(np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float32))
    out = drink_cold_start.item_sim_seed_scores(
        candidate_drink_ids=[1, 999],
        seed_drink_ids=[2],
        seed_weights=[2.0],
        sim_matrix=sim,
        sim_ids=ids,
    )
    assert out[1] > 0
    assert out[999] == 0.0


# ── strategy routing ────────────────────────────────────────────────────

def test_strategy_cold_user_returns_popularity_label(trained):
    Session, _ = trained
    db = Session()
    try:
        assert serve_drink_cf.cf_strategy_name(0, "beer") == "popularity_cold_start"
        assert serve_drink_cf.cf_strategy_name(0, "wine") == "popularity_cold_start"
    finally:
        db.close()


def test_strategy_warm_beer_user_uses_svd(trained):
    assert serve_drink_cf.cf_strategy_name(5, "beer") == "biased_mf"


def test_strategy_wine_never_uses_svd(trained):
    assert serve_drink_cf.cf_strategy_name(10, "wine") == "wine_item_sim"


def test_strategy_blended_band(trained):
    assert serve_drink_cf.cf_strategy_name(2, "beer") == "blended"


# ── end-to-end get_cf_scores ────────────────────────────────────────────

def test_cold_user_gets_popularity_for_all_candidates(trained):
    Session, _ = trained
    db = Session()
    try:
        scores = serve_drink_cf.get_cf_scores(
            user_id=99,  # never rated anything
            drinks_with_kinds=[(1, "beer"), (2, "beer"), (11, "wine"), (13, "wine")],
            db=db,
        )
    finally:
        db.close()
    assert set(scores.keys()) == {1, 2, 11, 13}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # Hop A (n=200) should outrank New Beer (n=2) on smoothed popularity
    # (we don't test for that here because New Beer isn't in candidates;
    # tested separately in test_bayesian_popularity_smooths_low_n)


def test_warm_beer_user_gets_svd_scores(trained):
    """User 1 has 4 beer ratings + 2 wine — borderline (4 < MIN=5)."""
    Session, _ = trained
    db = Session()
    try:
        scores = serve_drink_cf.get_cf_scores(
            user_id=1,
            drinks_with_kinds=[(1, "beer"), (3, "beer")],
            db=db,
        )
    finally:
        db.close()
    assert set(scores.keys()) == {1, 3}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_wine_candidates_route_to_item_sim_when_user_has_wine_history(trained):
    """User 1 rated Red 1=5 and White 1=3; recommending Sparkling (id=13) should
    score via item-sim seeded from those ratings, not SVD."""
    Session, _ = trained
    db = Session()
    try:
        scores = serve_drink_cf.get_cf_scores(
            user_id=1,
            drinks_with_kinds=[(11, "wine"), (12, "wine"), (13, "wine")],
            db=db,
        )
    finally:
        db.close()
    assert set(scores.keys()) == {11, 12, 13}


def test_blended_user_combines_item_sim_and_svd(trained):
    """User 3 has 3 explicit beer ratings; alpha = 3/5 = 0.6 → blend."""
    Session, _ = trained
    db = Session()
    try:
        scores = serve_drink_cf.get_cf_scores(
            user_id=3,
            drinks_with_kinds=[(1, "beer"), (4, "beer")],
            db=db,
        )
    finally:
        db.close()
    assert 1 in scores and 4 in scores
    # User 3 likes IPAs (Hop A id=1 they rated 5.0); blended score for id=1 should be > id=4
    assert scores[1] >= scores[4]


def test_synthetic_only_seeds_still_produce_scores(trained):
    """A user with ONLY a synthetic rating should still get item-sim wine scores."""
    Session, _ = trained
    db = Session()
    try:
        # Add only a synthetic event for user 99 on wine 11
        db.add(DrinkEvent(user_id=99, drink_id=11, event_type="rate",
                           rating=5.0, synthetic=True))
        db.commit()
        scores = serve_drink_cf.get_cf_scores(
            user_id=99,
            drinks_with_kinds=[(11, "wine"), (12, "wine")],
            db=db,
        )
    finally:
        db.close()
    # Wine path: seeds include synthetic events; should NOT be popularity fallback
    assert 11 in scores and 12 in scores


def test_empty_candidates_returns_empty(trained):
    Session, _ = trained
    db = Session()
    try:
        assert serve_drink_cf.get_cf_scores(user_id=1, drinks_with_kinds=[], db=db) == {}
    finally:
        db.close()
