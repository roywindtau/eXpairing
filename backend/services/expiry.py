"""
expiry.py
---------
Computes urgency scores for pantry items based on days remaining until expiry.

ROLE IN THE SYSTEM
------------------
Expiry urgency is a DOMAIN ADJUSTMENT in the CF-first scoring formula:

    final_score = γ·CF(user,recipe)       ← base preference prediction
                + δ·CB(pantry, recipe)    ← ingredient profile boost
                + α·expiry_urgency        ← domain: waste minimization ✓
                + β·match_ratio           ← domain: availability

Expiry urgency does not predict user preference — it adjusts rankings
so that recipes using soon-to-expire ingredients are surfaced first,
regardless of predicted preference strength.

FORMULA
-------
    urgency(days) = exp(-k · max(days, 0))    k = ln(2) / half_life

Default half_life = 3 days:
    0 days  → 1.00  (expires today — maximum urgency)
    3 days  → 0.50  (half-life point)
    6 days  → 0.25
    14 days → 0.04

Expired items (negative days) also return 1.0 — maximally urgent.
"""

import math
from datetime import date, datetime
from typing import Union

DEFAULT_HALF_LIFE_DAYS: float = 3.0


def days_until_expiry(expiry_date: Union[date, str]) -> int:
    """
    Return integer days between today and expiry_date.
    Negative means already expired.
    """
    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
    return (expiry_date - date.today()).days


def urgency_score(
    expiry_date: Union[date, str],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """
    Compute a [0,1] urgency score for a single pantry item.

    Expired items return 1.0 — they are maximally urgent
    (use them or discard them immediately).
    """
    days = days_until_expiry(expiry_date)
    k    = math.log(2) / half_life_days
    return round(math.exp(-k * max(days, 0)), 6)


def pantry_urgency_map(
    pantry_items: list[dict],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> dict[str, float]:
    """
    Return {ingredient_name -> urgency_score} for all pantry items.

    If an ingredient appears multiple times (different products),
    the highest urgency wins — the most urgent instance drives ranking.

    Each item dict must have:
        "ingredient"  : str  — canonical name e.g. "milk"
        "expiry_date" : str  — ISO date "YYYY-MM-DD"
    """
    result: dict[str, float] = {}
    for item in pantry_items:
        name  = item["ingredient"].strip().lower()
        score = urgency_score(item["expiry_date"], half_life_days)
        if name not in result or score > result[name]:
            result[name] = score
    return result
