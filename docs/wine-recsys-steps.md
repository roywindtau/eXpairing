# Wine Recommender — Implementation Steps (v1, completed)

This document is the **as-shipped record** of how the wine recommender was built — 11 steps, each independently shippable and tested. For the architecture and reasoning behind the design see [`wine-recsys-design.md`](./wine-recsys-design.md); for what to build next see [`wine-recsys-future.md`](./wine-recsys-future.md).

**Status:** ✅ All 11 steps complete. Backend test suite: 530 passing / 2 pre-existing recipe-CF failures unrelated to wine. Frontend `tsc --noEmit`: clean.

---

## Where things ended up vs. what was planned

Most of the plan landed as written. A few intentional divergences worth noting up-front:

| Topic | Plan | Reality | Why |
|---|---|---|---|
| Path B CF routing | "No SVD ever runs in Path B" — forced cold-start branch | Path B calls the same `get_cf_scores` as Path A | Hard-coding the cold-start branch would have ignored a warm user's explicit wine ratings in Path B, which is worse not better. The strategy matrix already does the right thing based on `n_explicit`. |
| Expert boost range | `[-0.25, +0.25]` (with negative penalties for bad pairs) | `[0, +0.25]` (positive boosts only) | Simpler, safer; a wrong negative penalty would actively hide a wine the user might love. Bad pairings just don't get the boost, which already deprioritizes them. |
| Score calibration | "z-score normalization" in some places | min-max across the candidate pool | Matches how the recipe stack's `services/scoring.py` actually calibrates. z-score was a doc slip. |
| Synthesizer order vs. scoring | Step 7 then Step 6 (scoring before synthesizer) | Step 6 then Step 7 (synthesizer before scoring) | Synthesizer is self-contained and has no upstream deps on scoring — easier to ship first. |
| Wine ratings storage | Originally "from CSV at request time" | In DB | Wines from X-Wines Test are ~1k rows — trivial to store, eliminates I/O at request time. |
| Dataset scale | Originally targeted X-Wines Slim (1k wines, 150k ratings) | Shipped on X-Wines Test (100 wines, 1k ratings) | Functional logic first; Slim/Full upgrade is a future-steps item (2.1). |

---

## Step 1 — Data + DB tables + seed scripts ✅

**Files:** `data/wine/download_wines.py`, `data/wine/clean_wines.py`, `backend/db/models.py` (added `Wine`, `WineEvent`), `backend/db/wine/seed_wines.py`, `backend/db/reset_wines.py`.

**Notes:** Wine data lives in SQLite. Wine users get `app_user_id = uid + 200_000`. The seeder is idempotent.

---

## Step 2 — Flavor bridge ✅

**Files:** `backend/ml/wine/serving/flavor_bridge.py`, `tests/wine/test_flavor_bridge.py`.

**Notes:** Lives in `backend/ml/wine/serving/`, not `backend/services/`, since it's a pure-Python lexicon transform with no side effects. `INGREDIENT_FLAVORS` has ~50 entries. Multi-word ingredients use substring matching.

---

## Step 3 — Wine CB (train + serve) ✅

**Files:** `backend/ml/wine/training/train_cb.py`, `backend/ml/wine/serving/serve_cb.py`, `tests/wine/test_wine_cb.py`.

**Notes:** Single TF-IDF model over a `style/variety + tokens` corpus, served via `cb_for_recipe(recipe)` / `cb_for_user(user_id, db)`. Artifacts: `drink_cb_matrix.npz`, `drink_cb_ids.npy`, `drink_cb_kinds.npy`, `drink_cb_vectorizer.pkl`, `drink_cb_meta.json`.

---

## Step 4 — Wine CF (train + item-sim + cold-start + serve) ✅

**Files:** `backend/ml/wine/training/train_wine_als.py`, `backend/ml/wine/training/item_similarity.py`, `backend/ml/wine/serving/cold_start.py`, `backend/ml/wine/serving/serve_cf.py`.

**Notes:** The wine CF model is confidence-weighted **ALS** (see `docs/wine-cf-experiments.md`); ALS beat Funk SVD on ranking metrics for the implicit-feedback wine data. At serve time, wine is too sparse for per-user matrix factorization, so `serve_cf.get_cf_scores` dispatches by `n_explicit`:

|  | wine candidate |
|---|---|
| 0 explicit | `bayesian_popularity` |
| ≥ 1 explicit | `item_sim_from_history` |

Item-sim is built with mean-centering and a ≥2-rating threshold (looser, for the tiny Test slice). The item-sim user-history seed includes synthetic events.

---

## Step 5 — Expert pairing rules ✅

**Files:** `backend/services/wine/expert_pairing.py`, `tests/wine/test_expert_pairing.py`.

**Notes:** Wine Harmonize CSV match (`WINE_BOOST_PER_MATCH = 0.10` per token overlap), capped at `MAX_BOOST = 0.25`. Boost is **non-negative** only. Path A only.

---

## Step 6 — Wine synthesizer + recipe-side hook ✅

**Files:** `backend/services/wine/synthesizer.py`, `tests/wine/test_wine_synthesizer.py`. Modified `backend/routers/recipes.py` with a one-line hook in `log_event`.

**Notes:** When `recipe_rating >= 4.0`, picks the top wines by CB + expert and writes `WineEvent(rating=4.0, synthetic=True)` rows. Guardrails: deduplicates per `(user, wine)`, never overwrites an explicit rating, fail-soft (swallows exceptions so a synthesizer bug never blocks recipe logging). Kill switch: `ENABLE_SYNTHETIC_WINE_RATINGS`.

---

## Step 7 — Wine scoring service ✅

**Files:** `backend/services/wine/scoring.py`, `tests/wine/test_wine_scoring.py`.

**Notes:** `rank_wines_for_recipe` (Path A: `0.45·cb + 0.25·cf + 0.20·expert + 0.10·prior`) and `rank_wines_for_user` (Path B: `0.55·cb + 0.30·cf + 0.15·prior`). Min-max calibration per component across the candidate pool. No DB queries — pure function over pre-computed signal dicts, easy to unit-test.

---

## Step 8 — Wine router ✅

**Files:** `backend/routers/drinks.py`, `tests/wine/test_wine_router.py`. Modified `backend/main.py` with `app.include_router(drinks.router)`.

**Endpoints:**
- `GET  /drinks/ranked?user_id=&top_n=` — Path B
- `GET  /drinks/pairings/{recipe_id}?user_id=&top_n=` — Path A
- `GET  /drinks/search?q=&limit=` — browse
- `GET  /drinks/{drink_id}` — detail
- `POST /drink-events` — rate (synthesizer hook is on **recipe** ratings only, not wine ratings)

**Two-stage pipeline:** Stage 1 orders by Bayesian-smoothed popularity to ≤ 2000 candidates; Stage 2 calls `rank_wines_*`.

**Test gotcha worth remembering:** `from backend.ml.wine.serving.serve_cb import cb_for_recipe` binds the function at import time, so tests must patch `backend.routers.drinks.cb_for_recipe` (the router's local reference), not the source module. See `tests/wine/test_wine_router.py` lines 76–80. Future-steps item 1.3 fixes this properly.

Also: `sqlite:///:memory:` creates a fresh DB per connection. Tests use `StaticPool` so every Session shares one connection.

---

## Step 9 — Frontend: Path B "Drinks For You" page ✅

**Files:** `frontend/src/api/drinks.ts`, `frontend/src/components/DrinkCard.tsx`, `frontend/src/pages/DrinksForYouPage.tsx`. Modified `frontend/src/App.tsx` to register the `/drinks` route and nav link.

**Notes:** Mirrors `RecipeFeedPage`. Sort dropdown (Total / Taste / Crowd / Popularity), CF-strategy banner (cold vs warm). Each card shows a `whyForYou()` reason line ("🍽️ Matches your food taste", "🤝 Similar to drinks you've liked", etc.) — translates the dominant signal into plain English so the algorithm name never leaks to the UI.

Wine ratings flow through `POST /drink-events`. No skip event (wine ratings are pure positive signal; dismiss is client-side only).

---

## Step 10 — Frontend: Path A pairing panel ✅

**Files:** `frontend/src/components/DrinkPairingPanel.tsx`. Modified `RecipeDetailPage.tsx` to accept `userId` and mount the panel.

**Notes:** Compact card grid (4–6 picks fit on screen), a `pairingReason()` helper that surfaces expert hits as `🎯 Harmonizes with Beef, Lamb, Grilled`. Inline 5-star rating per card. No "dismiss" in this context.

---

## Step 11 — Pipeline + docs ✅

**Files:** Updated `train_pipeline.sh` with 6 drink stages (D1–D6) plus `--skip-drinks` / `--drinks-only` flags. Updated `README.md` with a real drink-recommender section. Created this file pair (`wine-recsys-design.md`, `wine-recsys-steps.md`, and now `wine-recsys-future.md`).

---

## v1 acceptance checklist (all met)

- ✅ All recipe tests that were passing before this work are still passing.
- ✅ 156 new drink tests across 8 new test files, all passing.
- ✅ `python -m uvicorn backend.main:app --reload --port 8000` starts cleanly.
- ✅ `cd frontend && npm run dev` opens an SPA with a "Drinks" nav link.
- ✅ Opening a recipe shows a working pairing panel.
- ✅ Rating a recipe ≥ 4.0 in the UI then opening the Drink feed shows personalized results (synthesizer + Path B).
- ✅ Frontend `tsc --noEmit` passes; all UI surfaces have explainability copy (no raw algorithm names leak to the user).
