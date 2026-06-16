"""
synthesizer.py
--------------
Bootstraps a user's wine history from their RECIPE rating signal.

Why this exists
---------------
Without it, the Path-B "Drinks For You" flow has nothing to personalize on
until the user starts rating wines directly — which they may never do.
By writing a small set of synthetic WineEvent rows every time a user rates
a recipe ≥ 4.0, we get a sketch of their wine taste from day one.

Three guardrails prevent feedback loops (enforced elsewhere):
  1. `synthetic=True` flag → excluded from the wine CF training and
     item-similarity build. The model never learns from inferred data.
  2. Synthetic events ARE used as seeds for the item-sim path at serve
     time (see serve_cf._user_seed_wines). That's the intended effect —
     get personalization without contaminating the matrix.
  3. Explicit wine ratings supersede synthetic ones on the same
     (user, wine) pair — `_insert_if_no_existing` enforces this.

How it scores candidates
------------------------
A wine's "this user would probably like" score is:

    combined = cb_score + expert_boost

  - cb_score: cosine between the recipe's flavor-bridge doc and the
    wine's TF-IDF vector (from serve_cb). Captures the broad style
    affinity ("seafood-y recipe → seafood-y wines").
  - expert_boost: classic sommelier rules (Harmonize match + style
    heuristics). Adds crisp pairing knowledge the CB cosine alone
    might smear over.

If CB artifacts aren't loaded yet, the synthesizer falls back to
expert-boost only. If neither produces any positive scores, no
synthetic events are written.

Failure mode
------------
Wrapped in try/except — a failed synthesis MUST NOT break the recipe
rating UX. The exception is printed for debugging but swallowed.
"""

from __future__ import annotations

from typing import Optional

from backend.db.models import Wine, WineEvent, Recipe

# ── tunable constants (module-level so tests can patch them) ────────────

ENABLE_SYNTHETIC_WINE_RATINGS = True   # kill switch
SYNTHESIZE_THRESHOLD           = 4.0    # min recipe rating that triggers synthesis
SYNTHETIC_RATING               = 4.0    # value written into WineEvent.rating
N_SYNTHETIC                    = 3      # up to 3 wines per fire
CANDIDATE_POOL_SIZE            = 100    # pre-filter to top-N by CB before expert
MIN_COMBINED_SCORE             = 0.05   # ignore wines with essentially-zero affinity


def _candidate_wines(
    recipe: Recipe,
    db,
    n: int = CANDIDATE_POOL_SIZE,
) -> tuple[list[int], dict[int, float]]:
    """
    Return (top_n_wine_ids, cb_scores_dict).
    Uses CB cosine to pre-filter; if CB is unavailable, returns the
    n most-popular wines with cb_scores all = 0.
    """
    from backend.ml.wine.serving import serve_cb

    cb_scores: dict[int, float] = {}
    if serve_cb.model_available():
        cb_scores = serve_cb.cb_for_recipe(recipe)

    if cb_scores:
        top_ids = sorted(cb_scores.keys(), key=lambda d: -cb_scores[d])[:n]
        return top_ids, cb_scores

    # Fallback when CB unavailable: top-N by Bayesian-smoothed popularity.
    bayesian = (
        (Wine.avg_rating * Wine.n_ratings + 3.5 * 5)
        / (Wine.n_ratings + 5)
    )
    rows = (
        db.query(Wine.id)
        .filter(Wine.n_ratings.isnot(None))
        .order_by(bayesian.desc().nullslast())
        .limit(n)
        .all()
    )
    return [int(r[0]) for r in rows], {}


def _insert_if_no_existing(
    db,
    user_id: int,
    wine_id: int,
    rating: float,
) -> bool:
    """
    Insert a synthetic WineEvent unless one already exists for this
    (user, wine) — explicit or synthetic. Returns True if inserted.

    Skipping when ANY rating exists is intentional:
      - Skip explicit → never overwrite real user preference
      - Skip synthetic → don't accumulate duplicate inferences each time
                         the user rates another similar recipe
    """
    existing = (
        db.query(WineEvent.id)
        .filter(WineEvent.user_id    == user_id)
        .filter(WineEvent.wine_id    == wine_id)
        .filter(WineEvent.event_type == "rate")
        .filter(WineEvent.rating.isnot(None))
        .first()
    )
    if existing is not None:
        return False
    db.add(WineEvent(
        user_id=user_id,
        wine_id=wine_id,
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

    Returns the number of synthetic WineEvent rows written (0 if the
    feature is disabled, the rating is below threshold, the recipe is
    unknown, or all top candidates already have ratings).

    Fail-soft: any exception is caught, printed, and swallowed so the
    caller's recipe-rating transaction always succeeds.
    """
    if not ENABLE_SYNTHETIC_WINE_RATINGS:
        return 0
    if rating is None or rating < SYNTHESIZE_THRESHOLD:
        return 0

    try:
        # Local imports keep the synthesizer cheap to import (no heavy
        # ML deps unless this function actually fires).
        from backend.services.wine.expert_pairing import expert_boost_batch

        recipe = db.get(Recipe, recipe_id)
        if recipe is None:
            return 0

        n_inserted = 0
        top_ids, cb_scores = _candidate_wines(recipe, db)
        if top_ids:
            top_wines = (
                db.query(Wine)
                .filter(Wine.id.in_(top_ids))
                .all()
            )
            expert_scores = expert_boost_batch(recipe, top_wines)

            scored = [
                (w.id, cb_scores.get(w.id, 0.0) + expert_scores.get(w.id, 0.0))
                for w in top_wines
            ]
            scored.sort(key=lambda x: -x[1])

            picks = [wid for wid, s in scored
                     if s >= MIN_COMBINED_SCORE][:N_SYNTHETIC]
            for wid in picks:
                if _insert_if_no_existing(db, user_id, wid, SYNTHETIC_RATING):
                    n_inserted += 1

        if n_inserted > 0:
            db.commit()
        return n_inserted

    except Exception as exc:
        # Never break the caller's transaction. Log + roll back our own writes.
        print(f"[wine_synthesizer] failed for user={user_id} recipe={recipe_id}: {exc}")
        try:
            db.rollback()
        except Exception:
            pass
        return 0
