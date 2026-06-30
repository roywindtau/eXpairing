"""
preference_profile.py
---------------------
Wine cold-start onboarding: turn a new user's everyday-fruit picks into a
content-based (CB) taste profile.

Rationale
---------
A brand-new user has no wine ratings, so neither CF (needs >=5 ratings) nor the
CB taste-profile (built from rated wines) can say anything. But wines ARE
described by fruit notes, so a user's favorite fruits are a natural proxy for
wine taste. We hand-author a small mapping fruit -> wine character (grapes,
body, acidity, styles), INFER one details dict from the picks, and store it on
the User. At ranking time those details become a "seed" vector in the SAME CB
feature space as `train_cb.py`, blended into the CB profile as a decaying term.

Only the CB path uses this — CF and popularity are untouched.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

# ── ordinal maps (MUST match backend/ml/wine/training/train_cb.py) ───────────
ACIDITY = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
BODY = {"Very light-bodied": 0.0, "Light-bodied": 0.25, "Medium-bodied": 0.5,
        "Full-bodied": 0.75, "Very full-bodied": 1.0}
ABV_NEUTRAL = 0.5   # fruit gives no alcohol signal -> neutral mid of [0,1]

# ── hand-authored fruit -> wine character ────────────────────────────────────
# Grape names use standard X-Wines spellings; build_seed_vector() silently skips
# any grape not present in the trained grape_vocab (retraining-safe).
FRUIT_PROFILES: dict[str, dict] = {
    "lemon":      {"grapes": ["Sauvignon Blanc", "Riesling"],            "acidity": "High",   "body": "Light-bodied",  "styles": ["White", "Sparkling"]},
    "orange":     {"grapes": ["Riesling", "Viognier", "Gewürztraminer"], "acidity": "High",   "body": "Medium-bodied", "styles": ["White"]},
    "apple":      {"grapes": ["Chardonnay", "Pinot Grigio"],             "acidity": "High",   "body": "Light-bodied",  "styles": ["White", "Sparkling"]},
    "pear":       {"grapes": ["Chardonnay", "Pinot Grigio"],             "acidity": "Medium", "body": "Light-bodied",  "styles": ["White"]},
    "grapes":     {"grapes": ["Muscat Blanc", "Chardonnay"],            "acidity": "Medium", "body": "Medium-bodied", "styles": ["White", "Sparkling"]},
    "peach":      {"grapes": ["Viognier", "Chardonnay", "Riesling"],     "acidity": "Medium", "body": "Medium-bodied", "styles": ["White", "Rosé"]},
    "apricot":    {"grapes": ["Viognier", "Riesling"],                   "acidity": "Medium", "body": "Medium-bodied", "styles": ["White"]},
    "strawberry": {"grapes": ["Grenache", "Pinot Noir"],                 "acidity": "Medium", "body": "Light-bodied",  "styles": ["Rosé", "Red"]},
    "cherry":     {"grapes": ["Pinot Noir", "Merlot"],                   "acidity": "Medium", "body": "Medium-bodied", "styles": ["Red"]},
    "raspberry":  {"grapes": ["Pinot Noir", "Grenache"],                 "acidity": "Medium", "body": "Light-bodied",  "styles": ["Red", "Rosé"]},
    "plum":       {"grapes": ["Merlot", "Malbec"],                       "acidity": "Medium", "body": "Full-bodied",   "styles": ["Red"]},
    "blackberry": {"grapes": ["Cabernet Sauvignon", "Syrah/Shiraz", "Malbec"], "acidity": "Medium", "body": "Full-bodied", "styles": ["Red"]},
}

# The exact set the UI offers (insertion order preserved for display).
SUPPORTED_FRUITS: list[str] = list(FRUIT_PROFILES.keys())


def _modal(values: list[str]) -> str | None:
    """Most common label; ties broken by first appearance (Counter is stable)."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def infer_details(fruits: list[str]) -> dict:
    """
    Combine the selected fruits into ONE wine-taste details dict.

    Returns {"fruits", "grapes", "body", "acidity", "styles"} where grapes/styles
    are de-duplicated unions (order-preserving) and body/acidity are the modal
    label across the picks. Unknown fruit names are ignored. Empty input -> empty
    details (caller treats as "no prefs").
    """
    picks = [f.strip().lower() for f in (fruits or []) if f and f.strip()]
    known = [f for f in picks if f in FRUIT_PROFILES]
    if not known:
        return {}

    grapes: list[str] = []
    styles: list[str] = []
    bodies: list[str] = []
    acidities: list[str] = []
    for f in known:
        p = FRUIT_PROFILES[f]
        for g in p["grapes"]:
            if g not in grapes:
                grapes.append(g)
        for s in p["styles"]:
            if s not in styles:
                styles.append(s)
        bodies.append(p["body"])
        acidities.append(p["acidity"])

    return {
        "fruits":  known,
        "grapes":  grapes,
        "body":    _modal(bodies),
        "acidity": _modal(acidities),
        "styles":  styles,
    }


def inferred_styles(details: dict | None) -> set[str]:
    """Styles implied by the stored details — used as the CB style hard-filter."""
    if not details:
        return set()
    return {s for s in details.get("styles", []) if s}


def build_seed_vector(details: dict | None, blocks: dict | None) -> np.ndarray | None:
    """
    Turn stored taste details into ONE unweighted CB vector matching the trained
    matrix layout (same construction as train_cb.py; weights are applied later at
    serve time). Returns None if details/blocks are missing.

    `blocks` is serve_cb.get_blocks(): {"dim", "blocks", "grape_vocab"}.
    Grapes absent from grape_vocab are skipped, so a vocab change can't crash.
    """
    if not details or not blocks:
        return None

    dim = int(blocks["dim"])
    layout = blocks["blocks"]
    g_vocab = blocks.get("grape_vocab", {})
    vec = np.zeros(dim, dtype=np.float64)

    # grape multi-hot, block unit-normalized (1/sqrt(k)) like train_cb.py
    g_block = layout["grape"]
    cols = [g_block["start"] + g_vocab[g]
            for g in details.get("grapes", []) if g in g_vocab]
    if cols:
        v = 1.0 / np.sqrt(len(cols))
        for c in cols:
            vec[c] = v

    # ordinal/numeric scalars
    vec[layout["acidity"]["col"]] = ACIDITY.get(details.get("acidity"), 0.5)
    vec[layout["body"]["col"]]    = BODY.get(details.get("body"), 0.5)
    vec[layout["abv"]["col"]]     = ABV_NEUTRAL
    # region block left at zero — fruit gives no region signal.

    if not np.any(vec):
        return None
    return vec
