"""
ingredient_match.py
-------------------
Computes overlap between a user's pantry and a recipe's ingredient list.

ROLE IN THE SYSTEM: AVAILABILITY DOMAIN ADJUSTMENT
---------------------------------------------------
    final_score = γ·CF  + δ·CB  + α·expiry  + β·match_ratio  ←

match_ratio penalizes recipes requiring ingredients the user must buy.
β is per-user and learned from revealed cooking behavior via beta_updater.py.

MATCHING ALGORITHM
------------------
Three-pass approach to avoid both false positives and false negatives:

Pass 1 — Exact match (fast path)
    "eggs" == "eggs"  →  match

Pass 2 — Head-noun word-boundary match with compound exclusions
    "milk"   in "whole milk"      →  match  (milk is head noun)
    "butter" in "peanut butter"   →  NO match  (compound exclusion)
    "egg"    in "eggplant"        →  NO match  (not a word boundary)
    "tomato" in "cherry tomatoes" →  match  (singular of head noun)

    COMPOUND_EXCLUSIONS prevents false positives where the pantry
    ingredient is the head noun but the compound is a different product:
        butter ≠ peanut butter
        oil    ≠ essential oil
        cream  ≠ ice cream (when matching standalone "cream")

Pass 3 — token_set_ratio fallback (length-gated)
    Only fires when the pantry term is ≥55% as long as the recipe term,
    which prevents short terms matching long compound strings.

This replaces the previous fuzz.partial_ratio which caused "corn" to
incorrectly match "peppercorns" (partial_ratio=100 because "corn" is
literally contained in "peppercorns" as a substring).
"""

import re
from rapidfuzz import fuzz

FUZZY_THRESHOLD = 80

# Compound exclusions: (head_noun, disqualifying_qualifier) pairs.
# When a recipe ingredient has the head noun AND the qualifier,
# the pantry ingredient should NOT match.
# e.g. pantry="butter", recipe="peanut butter" → (butter, peanut) → no match
COMPOUND_EXCLUSIONS: frozenset[tuple[str, str]] = frozenset({
    ("butter", "peanut"),
    ("butter", "nut"),
    ("butter", "cocoa"),
    ("oil",    "essential"),
    ("oil",    "baby"),
    ("cream",  "ice"),
})

# Qualifiers that do NOT change the identity of an ingredient.
# pantry="butter" + recipe="unsalted butter" → still a match.
SAFE_QUALIFIERS: frozenset[str] = frozenset({
    "whole", "skim", "semi", "fresh", "frozen", "dried", "ground",
    "chopped", "diced", "sliced", "grated", "crushed", "minced",
    "large", "small", "medium", "baby", "wild", "organic", "raw",
    "cherry", "sweet", "plain", "unsalted", "salted", "light",
    "heavy", "all", "purpose", "extra", "virgin", "soft", "firm",
})


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _singularize(word: str) -> str:
    """Simple English singularizer for common ingredient plurals."""
    if word.endswith("oes") and len(word) > 4: return word[:-2]
    if word.endswith("ies") and len(word) > 4: return word[:-3] + "y"
    if word.endswith("es")  and len(word) > 4: return word[:-1]
    if word.endswith("s")   and len(word) > 3: return word[:-1]
    return word


def ingredient_matches(pantry_ing: str, recipe_ing: str) -> bool:
    """
    Return True if the pantry ingredient covers the recipe ingredient.

    Three-pass algorithm — see module docstring for full explanation.
    """
    p = _normalize(pantry_ing)
    r = _normalize(recipe_ing)

    # Pass 1: exact
    if p == r:
        return True

    r_words   = r.split()
    p_variants = {p, _singularize(p)}

    compound_excluded = False  # True when a variant matched but was blocked by COMPOUND_EXCLUSIONS
    for p_var in p_variants:
        # Find p_var as a complete word in recipe (exact or singular)
        r_singular = [_singularize(w) for w in r_words]

        if p_var in r_words:
            idx = r_words.index(p_var)
        elif p_var in r_singular:
            idx = r_singular.index(p_var)
        else:
            continue

        # Compound exclusion check
        preceding = set(r_words[:idx])
        if any((p_var, q) in COMPOUND_EXCLUSIONS for q in preceding):
            compound_excluded = True  # found but blocked — don't let Pass 3 override
            continue

        # Head noun (last word) always matches
        if idx == len(r_words) - 1:
            return True

        # Non-head: all preceding qualifiers must be safe
        if preceding <= SAFE_QUALIFIERS:
            return True

    # Pass 3: token_set_ratio fallback (length-gated to prevent short mismatches).
    # Skipped when compound exclusion already blocked this pair — fuzzy must not
    # override an explicit exclusion (e.g. "cream" ≈ "ice cream" via token_set_ratio=100).
    if compound_excluded:
        return False
    len_ratio = len(p) / max(len(r), 1)
    if len_ratio >= 0.55:
        if fuzz.token_set_ratio(p, r) >= FUZZY_THRESHOLD:
            return True

    return False


def match_ingredients(
    recipe_ingredients: list[str],
    pantry_ingredients: list[str],
) -> dict:
    """
    Match a recipe's ingredient list against the user's pantry.

    Returns:
        {
          "match_ratio": float [0,1]  — fraction of recipe ingredients covered
          "matched":     list[str]    — covered ingredients
          "missing":     list[str]    — ingredients user must buy
          "total":       int
        }
    """
    if not recipe_ingredients:
        return {"match_ratio": 1.0, "matched": [], "missing": [], "total": 0}

    matched, missing = [], []
    for ing in recipe_ingredients:
        found = any(ingredient_matches(p, ing) for p in pantry_ingredients)
        (matched if found else missing).append(ing)

    total = len(recipe_ingredients)
    return {
        "match_ratio": round(len(matched) / total, 6) if total else 1.0,
        "matched":     matched,
        "missing":     missing,
        "total":       total,
    }


def expiry_weighted_match(
    recipe_ingredients: list[str],
    urgency_map: dict[str, float],
) -> float:
    """
    Score how well a recipe covers your urgently-expiring pantry items.

    Weights each matched ingredient by its expiry urgency score.
    A recipe using your milk (expiring tomorrow) scores higher than
    one using your pasta (6 months left), even with equal match ratios.

    Normalized by pantry size (not recipe length) so that a complex recipe
    using both your expiring garlic AND milk ranks higher than a simple
    2-ingredient recipe using only milk — even though the simple recipe has
    a higher fraction-of-ingredients-covered.

    Returns float [0,1].
    """
    if not recipe_ingredients or not urgency_map:
        return 0.0

    total_urgency = 0.0
    for ing in recipe_ingredients:
        for pantry_ing, urgency in urgency_map.items():
            if ingredient_matches(pantry_ing, ing):
                total_urgency += urgency
                break

    # Divide by number of expiring pantry items: answers "what fraction of
    # your expiring pantry does this recipe use?" rather than penalising
    # recipes that happen to have many total ingredients.
    return round(min(total_urgency / len(urgency_map), 1.0), 6)
