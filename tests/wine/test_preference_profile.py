"""
test_preference_profile.py
--------------------------
Unit tests for backend/services/wine/preference_profile.py — the hand-authored
fruit -> wine-taste mapping used for cold-start onboarding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from backend.services.wine.preference_profile import (
    ACIDITY,
    BODY,
    build_seed_vector,
    infer_details,
    inferred_styles,
)

# A tiny stand-in for serve_cb.get_blocks(): 2 grapes, 1 region, 3 scalars.
_STUB_BLOCKS = {
    "dim": 6,
    "blocks": {
        "grape":   {"start": 0, "end": 2},
        "region":  {"start": 2, "end": 3},
        "acidity": {"col": 3},
        "body":    {"col": 4},
        "abv":     {"col": 5},
    },
    "grape_vocab": {"Pinot Noir": 0, "Merlot": 1},
}


# ── infer_details ─────────────────────────────────────────────────────────

def test_infer_details_cherry_is_red():
    d = infer_details(["cherry"])
    assert "Red" in d["styles"]
    assert "Pinot Noir" in d["grapes"]
    assert d["fruits"] == ["cherry"]


def test_infer_details_lemon_is_white():
    d = infer_details(["lemon"])
    assert "White" in d["styles"]
    assert d["acidity"] == "High"


def test_infer_details_unions_and_dedups():
    d = infer_details(["cherry", "raspberry"])  # both map to Pinot Noir
    assert d["grapes"].count("Pinot Noir") == 1  # de-duplicated
    assert "Red" in d["styles"]


def test_infer_details_empty_and_unknown():
    assert infer_details([]) == {}
    assert infer_details(["dragonfruit", "  "]) == {}


def test_infer_details_is_case_insensitive():
    assert infer_details(["CHERRY"]) == infer_details(["cherry"])


# ── inferred_styles ───────────────────────────────────────────────────────

def test_inferred_styles():
    assert inferred_styles(infer_details(["cherry"])) == {"Red"}
    assert inferred_styles({}) == set()
    assert inferred_styles(None) == set()


# ── build_seed_vector ─────────────────────────────────────────────────────

def test_build_seed_vector_sets_expected_columns():
    d = infer_details(["cherry"])  # grapes Pinot Noir + Merlot (both in vocab)
    vec = build_seed_vector(d, _STUB_BLOCKS)
    assert vec is not None
    # grape columns set and block unit-normalized (1/sqrt(2))
    assert vec[0] > 0 and vec[1] > 0
    np.testing.assert_allclose(np.linalg.norm(vec[0:2]), 1.0, atol=1e-9)
    # region column untouched
    assert vec[2] == 0.0
    # scalars use the train_cb maps
    assert vec[3] == ACIDITY[d["acidity"]]
    assert vec[4] == BODY[d["body"]]
    assert 0.0 <= vec[5] <= 1.0  # abv neutral


def test_build_seed_vector_skips_unknown_grapes():
    # Zinfandel is not in the stub vocab -> grape block stays zero, no crash.
    details = {"grapes": ["Zinfandel"], "body": "Full-bodied", "acidity": "High"}
    vec = build_seed_vector(details, _STUB_BLOCKS)
    assert vec is not None
    assert vec[0] == 0.0 and vec[1] == 0.0
    assert vec[3] == ACIDITY["High"]      # scalars still set
    assert vec[4] == BODY["Full-bodied"]


def test_build_seed_vector_none_inputs():
    assert build_seed_vector(None, _STUB_BLOCKS) is None
    assert build_seed_vector(infer_details(["cherry"]), None) is None
