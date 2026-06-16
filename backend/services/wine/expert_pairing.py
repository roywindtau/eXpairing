"""
expert_pairing.py
-----------------
Rule-based pairing boost that encodes classic sommelier knowledge.

Sits *on top of* CB + CF. The final ranker (scoring.py, Step 7) does:

    final = wα·CF + wβ·CB + wγ·expert_boost + wδ·popularity

Expert rules are deterministic, fast, and require no training data — they
are what surfaces "steak → Malbec" even when the CB cosine is noisy
(e.g. tiny wine corpus) or CF has no warmth.

WINE_HARMONIZE_MATCH
--------------------
Every X-Wines row has a `harmonize_csv` like "Beef,Lamb,Grilled". We
tokenize the recipe via flavor_bridge, intersect with the wine's
harmonize set, and award WINE_BOOST_PER_MATCH per matching category.
This makes the X-Wines dataset's pairing knowledge directly actionable.

The final value is capped at MAX_BOOST so the expert layer can sharpen a
ranking but never dominate CB+CF.

This module has zero DB calls and zero ML dependencies; pure data.
"""

from __future__ import annotations

from typing import Iterable

from backend.ml.wine.serving.flavor_bridge import bridge_recipe_doc

# ── tunable constants ───────────────────────────────────────────────────

WINE_BOOST_PER_MATCH = 0.10
MAX_BOOST            = 0.25


# ── helpers ─────────────────────────────────────────────────────────────

def _recipe_tokens(recipe) -> set[str]:
    """All bridged tokens for this recipe (ingredient words + tag words + bridged flavors)."""
    doc = bridge_recipe_doc(recipe)
    return set(doc.split()) if doc else set()


def _harmonize_tokens(harmonize_csv: str | None) -> set[str]:
    """
    Tokenize X-Wines Harmonize values into the same lowercase vocabulary
    flavor_bridge uses on the recipe side. "Beef,Lamb,Grilled" → {beef, lamb, grilled}
    Multi-word categories like "Sweet Dessert" → {sweet, dessert}.
    """
    if not harmonize_csv:
        return set()
    out: set[str] = set()
    for cat in harmonize_csv.split(","):
        for word in cat.strip().lower().split():
            if word.isalpha():
                out.add(word)
    return out


# ── single-pair scoring ─────────────────────────────────────────────────

def expert_boost(recipe, wine) -> float:
    """
    Compute the rule-based pairing boost for ONE (recipe, wine) pair.

    Args:
        recipe:  Recipe ORM row or any object exposing .ingredients_csv + .tags_csv
        wine:    Wine ORM row with .harmonize_csv

    Returns:
        float in [0, MAX_BOOST]. 0 means no rule fired.
    """
    if recipe is None or wine is None:
        return 0.0

    recipe_tokens = _recipe_tokens(recipe)
    if not recipe_tokens:
        return 0.0

    boost = 0.0
    harmonize = _harmonize_tokens(getattr(wine, "harmonize_csv", None))
    if harmonize:
        n_matches = len(harmonize & recipe_tokens)
        boost += n_matches * WINE_BOOST_PER_MATCH

    return round(min(boost, MAX_BOOST), 6)


# ── batch scoring ───────────────────────────────────────────────────────

def expert_boost_batch(recipe, wines: Iterable) -> dict[int, float]:
    """
    Score many wines against ONE recipe efficiently — compute the recipe's
    bridged token set once and reuse for every wine.

    Returns:
        dict[wine_id, boost] (only includes wines with boost > 0)
    """
    if recipe is None:
        return {}
    recipe_tokens = _recipe_tokens(recipe)
    if not recipe_tokens:
        return {}

    out: dict[int, float] = {}
    for wine in wines:
        if wine is None:
            continue
        harmonize = _harmonize_tokens(getattr(wine, "harmonize_csv", None))
        if not harmonize:
            continue
        n_matches = len(harmonize & recipe_tokens)
        boost = n_matches * WINE_BOOST_PER_MATCH
        if boost > 0:
            out[wine.id] = round(min(boost, MAX_BOOST), 6)

    return out
