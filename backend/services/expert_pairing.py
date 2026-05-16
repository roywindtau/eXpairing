"""
expert_pairing.py
-----------------
Rule-based pairing boost that encodes classic sommelier / brewer knowledge.

Sits *on top of* CB + CF. The final ranker (drink_scoring.py, Step 7) does:

    final = wα·CF + wβ·CB + wγ·expert_boost + wδ·popularity

Expert rules are deterministic, fast, and require no training data — they
are what surfaces "steak → Malbec" and "chocolate → Stout" even when the
CB cosine is noisy (e.g. tiny wine corpus) or CF has no warmth.

Two layers
----------
1. WINE_HARMONIZE_MATCH
   Every X-Wines row has a `harmonize_csv` like "Beef,Lamb,Grilled". We
   tokenize the recipe via flavor_bridge, intersect with the wine's
   harmonize set, and award WINE_BOOST_PER_MATCH per matching category.
   This makes the X-Wines dataset's pairing knowledge directly actionable.

2. BEER_STYLE_RULES
   A small hand-coded table of (style_keyword, recipe_token, boost)
   triples — classic brewer pairings the CB model can't easily learn
   from style names alone.

The final value is capped at MAX_BOOST so the expert layer can sharpen a
ranking but never dominate CB+CF.

This module has zero DB calls and zero ML dependencies; pure data.
"""

from __future__ import annotations

from typing import Iterable

from backend.ml.flavor_bridge import bridge_recipe_doc

# ── tunable constants ───────────────────────────────────────────────────

WINE_BOOST_PER_MATCH = 0.10
MAX_BOOST            = 0.25

# ── beer rules ──────────────────────────────────────────────────────────
# Each rule: (style substrings, recipe token set, boost).
# A rule fires when ANY style substring is in the beer's style AND ANY
# recipe token is present in the bridged recipe doc.
BEER_STYLE_RULES: list[tuple[set[str], set[str], float]] = [
    # spicy heat ↔ hoppy bitterness cuts through capsaicin
    ({"ipa", "india pale ale"},
     {"spicy", "curry", "chili", "chile", "jalapeno", "indian"},
     0.10),
    # roasted malt ↔ chocolate / coffee
    ({"stout", "porter"},
     {"chocolate", "dessert", "sweet", "vanilla", "caramel"},
     0.10),
    # roasted-meat affinity for dark beers
    ({"stout", "porter"},
     {"beef", "grilled", "bbq", "smoky", "bacon"},
     0.05),
    # crisp light beers with light food
    ({"pilsner", "lager", "wheat"},
     {"fish", "seafood", "salad", "salads", "shrimp", "chicken", "light"},
     0.05),
    # sour beers with cheese / fruit
    ({"sour", "saison", "gose"},
     {"cheese", "fruit", "berry"},
     0.05),
    # belgian ales with rich / spiced
    ({"belgian", "dubbel", "tripel"},
     {"cheese", "savory", "italian"},
     0.05),
    # amber / red with bbq / grilled
    ({"amber", "red ale", "brown"},
     {"bbq", "grilled", "beef"},
     0.05),
]


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


def _style_lower(style: str | None) -> str:
    return (style or "").lower()


# ── single-pair scoring ─────────────────────────────────────────────────

def expert_boost(recipe, drink) -> float:
    """
    Compute the rule-based pairing boost for ONE (recipe, drink) pair.

    Args:
        recipe:  Recipe ORM row or any object exposing .ingredients_csv + .tags_csv
        drink:   Drink ORM row with .kind, .style (beer) or .harmonize_csv (wine)

    Returns:
        float in [0, MAX_BOOST]. 0 means no rule fired.
    """
    if recipe is None or drink is None:
        return 0.0

    recipe_tokens = _recipe_tokens(recipe)
    if not recipe_tokens:
        return 0.0

    boost = 0.0

    if drink.kind == "wine":
        harmonize = _harmonize_tokens(getattr(drink, "harmonize_csv", None))
        if harmonize:
            n_matches = len(harmonize & recipe_tokens)
            boost += n_matches * WINE_BOOST_PER_MATCH

    elif drink.kind == "beer":
        style = _style_lower(getattr(drink, "style", None))
        if style:
            for style_keys, recipe_keys, rule_boost in BEER_STYLE_RULES:
                if any(sk in style for sk in style_keys) and (recipe_keys & recipe_tokens):
                    boost += rule_boost

    return round(min(boost, MAX_BOOST), 6)


# ── batch scoring ───────────────────────────────────────────────────────

def expert_boost_batch(recipe, drinks: Iterable) -> dict[int, float]:
    """
    Score many drinks against ONE recipe efficiently — compute the recipe's
    bridged token set once and reuse for every drink.

    Returns:
        dict[drink_id, boost] (only includes drinks with boost > 0)
    """
    if recipe is None:
        return {}
    recipe_tokens = _recipe_tokens(recipe)
    if not recipe_tokens:
        return {}

    out: dict[int, float] = {}
    for drink in drinks:
        if drink is None:
            continue
        boost = 0.0

        if drink.kind == "wine":
            harmonize = _harmonize_tokens(getattr(drink, "harmonize_csv", None))
            if harmonize:
                n_matches = len(harmonize & recipe_tokens)
                boost = n_matches * WINE_BOOST_PER_MATCH

        elif drink.kind == "beer":
            style = _style_lower(getattr(drink, "style", None))
            if style:
                for style_keys, recipe_keys, rule_boost in BEER_STYLE_RULES:
                    if any(sk in style for sk in style_keys) and (recipe_keys & recipe_tokens):
                        boost += rule_boost

        if boost > 0:
            out[drink.id] = round(min(boost, MAX_BOOST), 6)

    return out
