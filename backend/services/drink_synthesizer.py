"""
drink_synthesizer.py
--------------------
Bootstraps a user's drink history from their RECIPE rating signal.

Why this exists
---------------
Without it, the Path-B "Drinks For You" flow has nothing to personalize on
until the user starts rating drinks directly — which they may never do.
By writing a small set of synthetic DrinkEvent rows every time a user rates
a recipe ≥ 4.0, we get a sketch of their drink taste from day one.

Three guardrails prevent feedback loops (enforced elsewhere):
  1. `synthetic=True` flag → excluded from train_drink_cf and
     drink_item_similarity. SVD never learns from inferred data.
  2. Synthetic events ARE used as seeds for the item-sim path at serve
     time (see serve_drink_cf._user_seed_drinks). That's the intended
     effect — get personalization without contaminating the matrix.
  3. Explicit drink ratings supersede synthetic ones on the same
     (user, drink) pair — `_insert_if_no_explicit` enforces this.

How it scores candidates
------------------------
A drink's "this user would probably like" score is:

    combined = cb_score + expert_boost

  - cb_score: cosine between the recipe's flavor-bridge doc and the
    drink's TF-IDF vector (from serve_drink_cb). Captures the broad
    style affinity ("seafood-y recipe → seafood-y drinks").
  - expert_boost: classic sommelier / brewer rules (Harmonize match +
    style heuristics). Adds crisp pairing knowledge the CB cosine alone
    might smear over.

If CB artifacts aren't loaded yet, the synthesizer falls back to
expert-boost only. If neither produces any positive scores for a kind,
no synthetic events are written for that kind.

Failure mode
------------
Wrapped in try/except — a failed synthesis MUST NOT break the recipe
rating UX. The exception is printed for debugging but swallowed.
"""

from __future__ import annotations

from typing import Optional

from backend.db.models import Drink, DrinkEvent, Recipe

# ── tunable constants (module-level so tests can patch them) ────────────

ENABLE_SYNTHETIC_DRINK_RATINGS = True   # kill switch
SYNTHESIZE_THRESHOLD           = 4.0    # min recipe rating that triggers synthesis
SYNTHETIC_RATING               = 4.0    # value written into DrinkEvent.rating
N_SYNTHETIC_PER_KIND           = 3      # 3 beer + 3 wine = up to 6 per fire
CANDIDATE_POOL_SIZE            = 100    # pre-filter to top-N by CB before expert
MIN_COMBINED_SCORE             = 0.05   # ignore drinks with essentially-zero affinity


def _candidate_drinks_for_kind(
    recipe: Recipe,
    kind: str,
    db,
    n: int = CANDIDATE_POOL_SIZE,
) -> tuple[list[int], dict[int, float]]:
    """
    Return (top_n_drink_ids, cb_scores_dict) for one kind.
    Uses CB cosine to pre-filter; if CB is unavailable, returns the
    n most-popular drinks of this kind with cb_scores all = 0.
    """
    from backend.ml import serve_drink_cb

    cb_scores: dict[int, float] = {}
    if serve_drink_cb.model_available():
        cb_scores = serve_drink_cb.cb_for_recipe(recipe, kind_filter=kind)

    if cb_scores:
        top_ids = sorted(cb_scores.keys(), key=lambda d: -cb_scores[d])[:n]
        return top_ids, cb_scores

    # Fallback when CB unavailable: top-N by Bayesian-smoothed popularity.
    bayesian = (
        (Drink.avg_rating * Drink.n_ratings + 3.5 * 5)
        / (Drink.n_ratings + 5)
    )
    rows = (
        db.query(Drink.id)
        .filter(Drink.kind == kind)
        .filter(Drink.n_ratings.isnot(None))
        .order_by(bayesian.desc().nullslast())
        .limit(n)
        .all()
    )
    return [int(r[0]) for r in rows], {}


def _insert_if_no_existing(
    db,
    user_id: int,
    drink_id: int,
    rating: float,
) -> bool:
    """
    Insert a synthetic DrinkEvent unless one already exists for this
    (user, drink) — explicit or synthetic. Returns True if inserted.

    Skipping when ANY rating exists is intentional:
      - Skip explicit → never overwrite real user preference
      - Skip synthetic → don't accumulate duplicate inferences each time
                         the user rates another similar recipe
    """
    existing = (
        db.query(DrinkEvent.id)
        .filter(DrinkEvent.user_id    == user_id)
        .filter(DrinkEvent.drink_id   == drink_id)
        .filter(DrinkEvent.event_type == "rate")
        .filter(DrinkEvent.rating.isnot(None))
        .first()
    )
    if existing is not None:
        return False
    db.add(DrinkEvent(
        user_id=user_id,
        drink_id=drink_id,
        event_type="rate",
        rating=rating,
        synthetic=True,
    ))
    return True


def maybe_synthesize_on_recipe_rating(
    user_id: int,
    recipe_id: int,
    rating: float,
    db,
) -> int:
    """
    Main entry point — called from the recipe-side log_event hook.

    Returns the number of synthetic DrinkEvent rows written (0 if the
    feature is disabled, the rating is below threshold, the recipe is
    unknown, or all top candidates already have ratings).

    Fail-soft: any exception is caught, printed, and swallowed so the
    caller's recipe-rating transaction always succeeds.
    """
    if not ENABLE_SYNTHETIC_DRINK_RATINGS:
        return 0
    if rating is None or rating < SYNTHESIZE_THRESHOLD:
        return 0

    try:
        # Local imports keep the synthesizer cheap to import (no heavy
        # ML deps unless this function actually fires).
        from backend.services.expert_pairing import expert_boost_batch

        recipe = db.get(Recipe, recipe_id)
        if recipe is None:
            return 0

        n_inserted = 0
        for kind in ("beer", "wine"):
            top_ids, cb_scores = _candidate_drinks_for_kind(recipe, kind, db)
            if not top_ids:
                continue

            top_drinks = (
                db.query(Drink)
                .filter(Drink.id.in_(top_ids))
                .all()
            )
            expert_scores = expert_boost_batch(recipe, top_drinks)

            scored = [
                (d.id, cb_scores.get(d.id, 0.0) + expert_scores.get(d.id, 0.0))
                for d in top_drinks
            ]
            scored.sort(key=lambda x: -x[1])

            picks = [did for did, s in scored
                     if s >= MIN_COMBINED_SCORE][:N_SYNTHETIC_PER_KIND]
            for did in picks:
                if _insert_if_no_existing(db, user_id, did, SYNTHETIC_RATING):
                    n_inserted += 1

        if n_inserted > 0:
            db.commit()
        return n_inserted

    except Exception as exc:
        # Never break the caller's transaction. Log + roll back our own writes.
        print(f"[drink_synthesizer] failed for user={user_id} recipe={recipe_id}: {exc}")
        try:
            db.rollback()
        except Exception:
            pass
        return 0
