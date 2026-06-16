# Wine Recommender — Design Document

This doc explains **what the wine recommender does, how the pieces fit together, and why each design decision was made**. It is the right starting point for a new developer joining the project.

For the chronological as-shipped record see [`wine-recsys-steps.md`](./wine-recsys-steps.md); for what to build next see [`wine-recsys-future.md`](./wine-recsys-future.md).

---

## 1. What this module does

Adds **wine recommendations** to Fridge2Fork in two contexts:

- **Path A — "Pair this with…"**: on any recipe detail page, a panel suggests 4–6 wines for that specific dish.
- **Path B — "Drinks For You"**: a standalone `/drinks` page ranks wines personalized to the user's food + drink history.

Both paths share the same underlying scoring machinery (CB + CF + popularity + optional expert) but blend the signals with different weights and skip the expert layer on Path B (no specific recipe → no rule to fire).

---

## 2. The fundamental problem and how we solve it

There is **no public dataset that maps recipes to wines at scale**. We have:

- Food.com recipes (231k recipes, 1.1M ratings) — ingredients, tags, ratings.
- X-Wines Test (GitHub) — 100 wines, 1k ratings, plus a `Harmonize` column listing food categories each wine pairs with (`"['Beef', 'Lamb', 'Grilled']"`).

Zero users overlap between the two datasets. So we cannot do cross-domain collaborative filtering (e.g. "users who like recipe X also like wine Y") directly.

**Our solution: a shared flavor vocabulary that lets the two sides talk.** A recipe becomes a bag of flavor tokens (`spicy`, `fatty`, `umami`, `light`, `bold`); a wine already speaks that language (because TF-IDF over `style + variety + grape + harmonize` produces similar tokens). Once both sides live in the same vector space, cosine similarity is a meaningful pairing score.

On top of that we add:
- **Expert rules** (Harmonize match) for knowledge ML can't learn.
- **A synthesizer** that bootstraps wine history from highly-rated recipes — so Path B feels personalized before the user has ever rated a wine.

---

## 3. The four signal sources, in plain language

Every wine recommendation is a weighted blend of these four scores, each min-max calibrated across the candidate pool before blending:

### 3.1 CB — content match
TF-IDF cosine between
- Path A: the recipe's bridged-flavor document
- Path B: a weighted aggregate of the user's recipe-rating history (each recipe's bridged document weighted by `rating - 3.0`)

…against each candidate wine's TF-IDF vector.

**Implementation:** `backend/ml/wine/training/train_cb.py` (trainer) and `backend/ml/wine/serving/serve_cb.py` (`cb_for_recipe`, `cb_for_user`). The flavor bridge that translates ingredients to wine-side tokens lives at `backend/ml/wine/serving/flavor_bridge.py`.

### 3.2 CF — collaborative filtering
Routed by `n_explicit_ratings`. The strategy matrix:

|  | wine candidate |
|---|---|
| 0 explicit | `bayesian_popularity` |
| ≥ 1 explicit | `item_sim_from_history` |

**Wine serving never uses per-user matrix factorization** — the X-Wines Test slice has too few ratings, and demo users won't accumulate enough explicit wine ratings to trigger warm CF anyway. (The offline wine CF model is confidence-weighted ALS, chosen over Funk SVD on ranking metrics — see `docs/wine-cf-experiments.md`.)

The user's "history" used to seed item-sim **includes synthetic events** (from the synthesizer) — but those events are **excluded from CF model training** so the latent factors aren't polluted with guesses.

**Implementation:** `backend/ml/wine/training/item_similarity.py`, `backend/ml/wine/serving/cold_start.py`, and the unifying serve layer `backend/ml/wine/serving/serve_cf.py` (`get_cf_scores` is the single entry point).

### 3.3 Expert boost — pairing wisdom ML can't learn (Path A only)
A bounded `[0, +0.25]` boost computed from:
- **Wine Harmonize match**: count overlaps between recipe-derived food tokens and the wine's `Harmonize` CSV. Each match adds `WINE_BOOST_PER_MATCH = 0.10`.

The boost is **always non-negative**: a bad pairing simply doesn't get the boost, which already deprioritizes the wine relative to a good pairing. We deliberately don't apply negative penalties (a wrong penalty would actively hide a wine the user might love).

**Implementation:** `backend/services/wine/expert_pairing.py` (`expert_boost` for one pair, `expert_boost_batch` for many wines against one recipe — the more common call site).

### 3.4 Popularity prior — tiebreaker
`avg_rating × log(1 + n_ratings)`. Carries the score for true new users with no signal at all, and prevents obscure wines from winning by accident on a single rating.

---

## 4. The two scoring formulas

### Path A (`rank_wines_for_recipe`)
```
final_A = 0.45·cb + 0.25·cf + 0.20·expert + 0.10·prior
```

CB dominates because content match is the most explainable signal in a pairing context — "this wine matches the recipe's flavor profile" is the answer the user is implicitly asking for.

### Path B (`rank_wines_for_user`)
```
final_B = 0.55·cb + 0.30·cf + 0.15·prior
```

No expert term (no specific recipe to pair against). The 0.20 weight Path A spent on expert redistributes to CB (now 0.55) and prior (now 0.15) — CB still dominates, because Path B's whole pitch is "we picked this from your food taste."

### Why min-max calibration
Each component is normalized to `[0, 1]` across the **candidate pool of the current request** before blending. Without this, one wide-ranging dimension (e.g. popularity prior, which spans orders of magnitude due to `log(n_ratings)`) would swamp a tightly clustered dimension (e.g. CB cosine, which usually lives in `[0, 0.3]` on real text). Same reasoning as `backend/services/scoring.py` for recipes.

**Implementation:** `backend/services/wine/scoring.py`. Pure-functional — accepts pre-computed signal dicts and returns ranked `WineScore` objects. No DB queries. Trivial to unit-test.

---

## 5. The knowledge layer (three modules unique to wine)

These are the only modules without a recipe-side equivalent. Everything else in the wine stack mirrors a file that already existed for recipes.

### 5.1 `backend/ml/wine/serving/flavor_bridge.py` — translation
A hand-curated `INGREDIENT_FLAVORS` lexicon (~50 entries) maps recipe ingredients to wine-side flavor tokens:

```
chili    → spicy, bold
beef     → heavy, umami, rich, fatty
lemon    → citrus, acidic, light
chocolate → sweet, rich, roasted
cream    → creamy, rich, smooth, fatty
```

Three exported helpers:
- `bridge_ingredients(csv) -> list[str]`
- `bridge_tags(csv) -> list[str]`
- `bridge_recipe_doc(recipe) -> str` — the canonical "give me this recipe as a flavor doc" function used by CB serving.

Multi-word ingredients are matched by substring (`"olive oil"` matches `"oil"`'s flavor tokens). Repetition is preserved so common ingredients carry more weight in the TF-IDF.

### 5.2 `backend/services/wine/expert_pairing.py` — rule-based pairing wisdom
Encodes the kind of pairing knowledge that pure cosine similarity can't surface: "fatty meat → tannic wine," "spicy curry → off-dry Riesling," "fish → high-acid white." Covered in §3.3.

### 5.3 `backend/services/wine/synthesizer.py` — cold-start bootstrap

When the user rates a recipe ≥ 4.0 stars, the synthesizer:
1. Picks the top CB+expert wine matches for that recipe (capped at 3 by default).
2. Writes `WineEvent(rating=4.0, synthetic=True)` rows for those wines.

This is the mechanism that makes Path B feel personalized after just a few recipe ratings — the user's food taste **leaks into a sketch of their wine taste** before they've explicitly rated a single wine.

**Three guardrails** prevent feedback loops and data quality issues:
1. **Synthetic rows are flagged** (`synthetic=True`) and **excluded from CF model training**. The matrix factorization never learns from data we invented.
2. **Synthetic rows feed only the item-sim seed at serve time** — they help retrieve "wines similar to ones we think you'd like" via *real* co-rating patterns from X-Wines.
3. **Explicit ratings always supersede synthetic ones** on the same `(user, wine)` pair. Once the user expresses real taste, the guess is dropped.

The synthesizer is **fail-soft**: any exception in synthesis is swallowed so it can never block the recipe-rating endpoint that triggered it. The kill switch is `ENABLE_SYNTHETIC_WINE_RATINGS = True` at the top of the module.

**Hook point:** one line at the end of `backend/routers/recipes.py::log_event`:
```python
if payload.event_type == "rate" and payload.rating is not None:
    maybe_synthesize_on_recipe_rating(payload.user_id, payload.recipe_id, payload.rating, db)
```

---

## 6. How the cold-start cascade works (Path B)

The recommendation degrades gracefully down a clear ladder:

1. **Brand-new user, zero recipes, zero wines.** Popularity prior carries; user sees the highest-rated wines overall.
2. **Some recipe history, no wine events.** `cb_user` lights up — the user's food taste drives picks via the flavor bridge.
3. **First "liked" recipe rating (≥ 4.0).** Synthesizer fires. Path-B CF now has seed events to compute item-sim against.
4. **User starts rating wines explicitly.** CF reweights toward explicit history via item-sim seeded from those ratings (wine stays on item-sim by design, see §3.2).

The same Path B endpoint serves all four stages — no special-casing.

---

## 7. Two-stage candidate-then-rank pipeline

Just like `backend/routers/recipes.py`, wine endpoints use two stages because scoring the full wine catalog per request would be wasteful:

**Stage 1 — Candidate generation.** Order by Bayesian-smoothed `(avg_rating, n_ratings)`, cap at 2000 candidates. SQL-side, fast.

**Stage 2 — Ranking.** Compute CB / CF / expert (Path A only) / prior for each candidate. Min-max calibrate. Blend per the formula. Sort by `final_score`. Return top N.

This keeps wine-endpoint latency comparable to the recipe endpoint.

---

## 8. Database schema additions

A single flat `wines` table plus a `wine_events` ratings table. (The module
originally used joined-table inheritance — a `Drink` base + `Wine` child with a
`kind` discriminator — to support multiple drink types; beer was dropped and the
schema collapsed to wine-only, removing the inheritance and the `kind` column.)

```python
class Wine(Base):
    __tablename__ = "wines"
    id                = Column(Integer, primary_key=True, index=True)
    name              = Column(String, nullable=False, index=True)
    producer          = Column(String, nullable=True)
    country           = Column(String, nullable=True)
    style             = Column(String, nullable=True)
    abv               = Column(Float, nullable=True)
    avg_rating        = Column(Float, nullable=True)
    n_ratings         = Column(Integer, default=0)
    harmonize_csv     = Column(Text, nullable=True)   # 'Beef,Lamb,Pasta'
    review_tokens_csv = Column(Text, nullable=True)
    grapes_csv        = Column(Text, nullable=True)
    body              = Column(String, nullable=True)   # 'Full-bodied' | 'Medium-bodied' | ...
    acidity           = Column(String, nullable=True)   # 'High' | 'Medium' | 'Low'
    region            = Column(String, nullable=True)

class WineEvent(Base):
    __tablename__ = "wine_events"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"),  nullable=False, index=True)
    wine_id     = Column(Integer, ForeignKey("wines.id"),  nullable=False, index=True)
    event_type  = Column(String, nullable=False)        # v1: 'rate' only
    rating      = Column(Float, nullable=True)
    synthetic   = Column(Boolean, default=False, nullable=False)
    created_at  = Column(DateTime, server_default=func.now(), index=True)
```

**Why one flat table** rather than the original inheritance: with a single drink
type, the `Drink` base + `Wine` child split and the `kind` discriminator added
indirection for no benefit — every query was already wine. Collapsing removes a
join and a discriminator column while keeping all the same fields.

**Schema migration:** dev data is disposable — `python -m backend.db.reset_wines`
drops and recreates `wines` + `wine_events`, then the wine seed repopulates. For
a fresh DB, `init_db()` (`Base.metadata.create_all`) is enough. The project
doesn't use Alembic.

---

## 9. API surface

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/drinks/ranked?user_id=&top_n=20`           | Path B feed |
| `GET`  | `/drinks/pairings/{recipe_id}?user_id=&top_n=10` | Path A panel |
| `GET`  | `/drinks/search?q=&limit=40`                | Browse / search (not personalized) |
| `GET`  | `/drinks/{drink_id}`                              | Wine detail |
| `POST` | `/drink-events`                                   | Rate a wine (v1: `event_type="rate"` only) |

The `/drinks/*` route prefix and the `drink_*` response field names are kept as a
stable API contract (the frontend depends on them); the catalog is wine-only.

The response shape for the two ranking endpoints (defined in `backend/routers/drinks.py::DrinkScoreOut`):
```python
class DrinkScoreOut(BaseModel):
    drink_id:     int
    drink_name:   str
    kind:         str
    final_score:  float
    cb_score:     float
    cf_score:     float
    expert_boost: float        # always 0.0 in Path B (no recipe to pair against)
    prior_score:  float
    cf_strategy:  str          # 'popularity_cold_start' | 'wine_item_sim'
                               #  — used internally for UI, never shown raw
    avg_rating:   float | None
    n_ratings:    int
    abv:          float | None
    producer:     str | None
    # wine-specific
    style:         str | None
    grapes_csv:    str | None
    harmonize_csv: str | None
```

**`POST /drink-events` deliberately has no synthesizer hook.** Synthesis is triggered only by recipe ratings — drink ratings are the user's explicit signal and feed directly into item-sim. Synthesizing more drink events from drink ratings would cause runaway loops.

---

## 10. The user-facing "why this wine" explanation

The UI never exposes raw algorithm names (`blended`, `wine_item_sim`) to the user. Instead, both Path A and Path B translate the dominant score component into a short sentence.

### Path A — `pairingReason()` in `DrinkPairingPanel.tsx`
| Condition | Sentence |
|---|---|
| `expert_boost > 0` | 🎯 Harmonizes with Beef, Lamb, Grilled |
| `cb_score` dominates | ✨ Flavor match |
| `cf_score` dominates | ✨ Loved by similar drinkers |
| Pure popularity wins | _(no line — no compelling story)_ |

### Path B — `whyForYou()` in `DrinkCard.tsx`
| Condition | Sentence |
|---|---|
| `cb_score` dominates | 🍽️ Matches your food taste |
| `cf_strategy == 'wine_item_sim'` | 🤝 Similar to wines you've liked |
| `cf_strategy == 'popularity_cold_start'` | 🔥 Loved by the community |
| Pure popularity wins | 🔥 Highly rated overall |

Both helpers are pure functions over the response payload — no extra API calls.

---

## 11. Why these design choices (and not alternatives)

### Why not real cross-domain CF?
Latent-space alignment between the recipe and wine spaces would require shared anchor entities (users who appear in both Food.com and X-Wines). We have zero. Latent alignment without anchors is research-grade and not what the v1 milestone needs.

### Why does Path B use the same CF function as Path A (not a forced cold-start)?
The original design said "no warm CF ever runs in Path B." On reflection that was over-restrictive — ignoring a user's explicit wine history in Path B makes recommendations *worse*, not better. The strategy matrix in `serve_cf.get_cf_scores` already does the right thing for both warm and cold users. What makes Path B *Path B* is the **CB source** (user history aggregate, not specific recipe) and the **weights** (no expert), encoded in `scoring.rank_wines_for_user`.

### Why no per-user matrix factorization at serve time?
X-Wines Test has ~1k ratings — far too sparse to serve meaningful per-user latent factors, and demo users won't accumulate enough explicit wine ratings to trigger warm CF anyway. So serving stays on Bayesian popularity → item-sim. (The offline wine CF model is confidence-weighted ALS — chosen over Funk SVD on ranking metrics, see `docs/wine-cf-experiments.md`.)

### Why no MMR diversity reranking in v1?
The recipe stack uses MMR (`λ=0.7`) to break up monotonous feeds. Wine deferred MMR to keep the v1 integration focused on correctness (data flow, CB/CF/expert blend, synthesizer loop) before adding diversity behavior. Listed in `wine-recsys-future.md` item 2.3.

### Why hand-curated rules instead of a learned pairing model?
Two reasons:
1. We don't have the training data for a learned model (no recipe ↔ wine co-rating signal).
2. The rules are **interpretable and editable in 5 minutes**. A learned model is neither. Until the lexicon becomes a real bottleneck, hand-curated wins.

---

## 12. Architecture diagram

```
                 
                
Offline data ────┤
                 │
                 └─── X-Wines Test (~1k ratings + Harmonize)
                              │
                              ▼
                 ┌────────────────────────┐
       Seed →    │ Wine rows (100 wines)  │   seed_wines.py
                 │ WineEvent rows         │
                 │ (synthetic=False)      │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
       Train →   │ models/                │   train_cb.py
                 │   wine_cb_*.npz/pkl   │   item_similarity.py
                 │   drink_sim_wine.npz   │   wine/train_wine_als.py
                 └────────────┬───────────┘
                              │
                              ▼
   ┌─────────────────── Knowledge layer (new) ───────────────────┐
   │  flavor_bridge.py     expert_pairing.py    synthesizer    │
   │  (translation)        (Harmonize + rules)  (write-side hook)│
   └────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
                 ┌────────────────────────┐
       Serve →   │ scoring.py       │   rank_wines_for_recipe (Path A)
                 │   min-max calibrate    │   rank_wines_for_user   (Path B)
                 │   + weighted blend     │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
        API →    │ /drinks/ranked         │   Path B feed
                 │ /drinks/pairings/{id}  │   Path A panel
                 │ /drinks/search         │   Browse
                 │ /drinks/{id}           │   Detail
                 │ /drink-events          │   Rate
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
        UI →     │ DrinksForYouPage.tsx   │   Path B (+ DrinkCard.tsx + whyForYou)
                 │ DrinkPairingPanel.tsx  │   Path A (mounted on RecipeDetailPage)
                 └────────────────────────┘
```

---

## 13. File layout — mirror table

The wine stack mirrors the existing recipe stack file-for-file. Three files are genuinely new (the knowledge layer).

| Existing recipe file | New wine file |
|---|---|
| `backend/db/models.py` (User, Recipe, UserEvent) | Same file: added `Wine`, `WineEvent` |
| `backend/db/seed_recipes.py` | `backend/db/wine/seed_wines.py` |
| `backend/ml/train_cb.py`      | `backend/ml/wine/training/train_cb.py` |
| `backend/ml/serve_cb.py`      | `backend/ml/wine/serving/serve_cb.py` |
| `backend/ml/train_cf.py`      | `backend/ml/wine/training/train_wine_als.py` (wine ALS) |
| `backend/ml/serve_cf.py`      | `backend/ml/wine/serving/serve_cf.py` |
| `backend/ml/item_similarity.py` | `backend/ml/wine/training/item_similarity.py` |
| `backend/ml/cold_start.py`    | `backend/ml/wine/serving/cold_start.py` |
| `backend/services/scoring.py` | `backend/services/wine/scoring.py` |
| `backend/routers/recipes.py`  | `backend/routers/drinks.py` |
| `frontend/src/api/client.ts`  | `frontend/src/api/drinks.ts` |
| `frontend/src/pages/RecipeFeedPage.tsx` | `frontend/src/pages/DrinksForYouPage.tsx` |
| `frontend/src/components/RecipeCard.tsx` | `frontend/src/components/DrinkCard.tsx` |
| _(no equivalent)_             | `frontend/src/components/DrinkPairingPanel.tsx` (Path A — new pattern) |

**Three files with no recipe equivalent** (the knowledge layer):
- `backend/ml/wine/serving/flavor_bridge.py`
- `backend/services/wine/expert_pairing.py`
- `backend/services/wine/synthesizer.py`

---

## 14. How to read this codebase (suggested order)

If you've never seen the wine module before, the fastest path to understanding is:

1. `docs/wine-recsys-design.md` — this file
2. `backend/db/models.py` — `Wine` + `WineEvent` classes (the data shape)
3. `backend/ml/wine/serving/flavor_bridge.py` — the lexicon (small, concrete, eye-opening)
4. `backend/services/wine/expert_pairing.py` — the rules
5. `backend/services/wine/scoring.py` — the formulas
6. `backend/routers/drinks.py` — how the endpoints glue everything together
7. `frontend/src/components/DrinkCard.tsx` and `DrinkPairingPanel.tsx` — how the UI explains itself
8. `tests/wine/test_wine_*` — runnable spec of the contract

Total reading time: ~45 minutes. After that you should be able to extend any piece confidently.
