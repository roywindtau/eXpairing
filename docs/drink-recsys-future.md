# Drink Recommender — Future Work

This document tracks **post-v1 improvements** to the drink recommender module. v1 is feature-complete and shipping (see [`drink-recsys-steps.md`](./drink-recsys-steps.md) for the 11 steps that built it, and [`drink-recsys-design.md`](./drink-recsys-design.md) for the architecture). Everything here is deferred — listed roughly by impact-per-effort.

The list is split into three buckets:

1. **Quick wins** — small, safe, high impact. Pick up first.
2. **Bigger features** — extend the v1 functionality in meaningful ways.
3. **Research / nice-to-have** — speculative or low-ROI; only if you have spare cycles.

---

## 1 · Quick wins

### 1.1 Train the recipe CF artifacts to clear the 2 pre-existing test failures

Not strictly drink work, but it's the only thing keeping `pytest` from showing 532/532 green:

```bash
python -m backend.ml.train_cf --no-implicit
python -m backend.ml.item_similarity
```

This produces `models/cf_meta.json` and `models/item_sim_matrix.npz`, which the two failing tests check for (`tests/test_scoring_edge_cases.py::test_cf_meta_records_no_implicit` and `tests/test_ml_behavior.py::test_cold_start_cf_scores_differentiated`).

### 1.2 Add tests for the Path-B "why this drink" explanation

`frontend/src/components/DrinkCard.tsx::whyForYou()` translates the dominant score component into a sentence ("Matches your food taste" / "Predicted from your drink ratings" / etc.). Pure function, ~30 lines, 8 cases — easy to unit test with Vitest if/when frontend tests get added. Currently uncovered.

### 1.3 Refactor router imports for testability

`backend/routers/drinks.py` uses `from backend.ml.serve_drink_cb import cb_for_recipe`, which binds the function at import time and broke our first attempt at router tests (we had to patch the router's local reference instead of the source module — documented in `tests/test_drinks_router.py` lines 76–80).

Cleaner pattern: `from backend.ml import serve_drink_cb` then call `serve_drink_cb.cb_for_recipe(...)`. Tests can then patch the source module directly and it just works. Same applies to `serve_drink_cf`, `expert_pairing`, `drink_scoring`.

### 1.4 Expand the `INGREDIENT_FLAVORS` lexicon

`backend/ml/flavor_bridge.py` currently has ~50 ingredient → flavor-token entries. Recipe coverage in the Food.com 20-row dev seed is decent but on the full 231k recipe corpus, many recipes will produce empty flavor docs (no entries match). Run a quick coverage analysis:

```python
from backend.ml.flavor_bridge import bridge_ingredients
# count how many ingredients across the corpus return empty
```

Add the top-50 missing ingredients by frequency (likely cilantro, paprika, coconut milk, soy sauce, etc.).

### 1.5 Add a `/drinks` browse page (Path-B sibling to `BrowsePage`)

The backend already exposes `GET /drinks/search?q=&kind=&limit=` and `getFrontend has the wrapper (`searchDrinks` in `api/drinks.ts`). All that's missing is a `DrinksBrowsePage.tsx` mirroring `BrowsePage.tsx` — a text-search + kind toggle UI for the full drink catalog. ~150 lines of React. Useful for the "find me an Allagash Tripel" flow.

### 1.6 "Reset my drink history" button on the profile page

The synthesizer accumulates synthetic events every time the user rates a recipe ≥ 4.0. If they change food preferences or want to start fresh, there's no UI to clear them. Add a `DELETE /drink-events/{user_id}?synthetic_only=true` endpoint + button in `ProfilePage.tsx`. ~30 lines.

---

## 2 · Bigger features

### 2.1 Upgrade X-Wines Test → Slim or Full

**Why:** Test gives us 100 wines + 1k ratings — enough to validate the pipeline but a thin demo catalog. Slim (~1k wines, ~150k ratings) and Full (~100k wines, ~21M ratings) are hosted on Google Drive (CC0 license).

**What to do:**
- Either add `gdown` to `requirements.txt` and extend `data/download_drinks.py` with `gdown.download()` calls for the Slim/Full Drive file IDs; or document a manual download step.
- Re-run `python -m backend.db.seed_drinks` and `seed_drink_ratings` (idempotent, but you may want to clear `drinks` and `drink_events` first to avoid mixing).
- Slim/Full include `Sweetness` and `Tannin` columns. Our schema already has nullable fields for them — populate during seed.
- The wine rules in `expert_pairing.py` already check `if drink.tannin is not None and drink.tannin >= 4:` etc. so they start firing automatically. **No code change needed.**
- Re-train: `python -m backend.ml.train_drink_cb && python -m backend.ml.drink_item_similarity`.

**Highest ROI of all the items here.** Unlocks a real wine catalog and activates dormant expert rules.

### 2.2 Wine & Food Pairing dataset (data-driven expert rules)

**Why:** Once v1.1 ships, we'll see whether the hand-coded rules + Harmonize match produce good pairings. If they feel weak for wines without `Harmonize` (e.g. a Greek wine with no curator notes), a dedicated pairing dataset is the right next step.

**What to add:**
- `data/wine_food_pairings.csv` from a Kaggle dataset or web-scraped curated source.
- New `backend/services/pairing_lookup.py` with `load_pairings()` (called at app startup) and `boost(recipe, drink) -> float in [0, +0.15]`.
- Extend `expert_pairing.expert_boost` to combine: `harmonize + lookup + rules`, still capped at `MAX_BOOST=0.25`.
- Tests in `tests/test_pairing_lookup.py`.

### 2.3 MMR diversity reranking for drinks

**Why:** Once enough users browse `/drinks`, we'll see if results feel repetitive (five IPAs in a row, three Malbecs). MMR fixes that. The recipe stack already does this; drinks deferred it for v1 to keep the first integration focused on correctness.

**What to add:**
- After scoring in `drink_scoring.rank_drinks_for_*`, apply MMR with `λ=0.7`. Diversity dimension: `(kind, style/variety)` Jaccard.
- Mirror `backend/services/scoring.py`'s MMR pass (~30 lines).
- Add tests confirming the top item still wins and the next-N varies in style.

### 2.4 Multi-event-types for drinks

**Why:** Once users engage with the drinks UI, they'll want to "save for later" or "skip" without rating. v1 only supports `event_type="rate"`.

**What to add:**
- Update `DrinkEvent.event_type` validation to accept `like`, `skip`, `save`.
- Update `POST /drink-events` schema validation.
- Update `serve_drink_cf._user_seed_drinks` to convert non-`rate` events into implicit ratings (`like → 4.0`, `save → 3.5`, `skip → 1.5`), mirroring how recipe CF handles `cook` events.
- Surface `Save` / `Skip` buttons on `DrinkCard.tsx` next to `Rate`.

### 2.5 Drink offline evaluation pipeline

**Why:** The recipe stack has `backend/ml/evaluate.py` that computes RMSE, Precision@K, Recall@K, NDCG@K, and ablation on a held-out set. Drinks have **no equivalent**. We're shipping v1 without quantitative ranking quality numbers.

**What to add:**
- `backend/ml/evaluate_drinks.py`. Hold out 20% of `DrinkEvent` rows (non-synthetic only), train, predict, score.
- Lifecycle simulation: NDCG@10 vs `n_explicit_drink_ratings` to validate the cold→warm ramp (popularity → item-sim → blend → SVD).
- Per-component ablation: CB-only / CF-only / expert-only / full hybrid.
- Add to `train_pipeline.sh` as a final stage.

### 2.6 `paired_recipe_id` analytics column

**Why:** When the synthesizer matures and we want to learn from real user pick-rates ("of the 6 drinks shown for Beef Bourguignon, which one did users actually rate?"), we need to record the pairing context.

**What to add:**
- Migration to add `paired_recipe_id` (nullable FK to `recipes.id`) to `DrinkEvent`.
- Capture in `POST /drink-events` when the request comes from the pairing panel (frontend adds `paired_recipe_id` to the payload when set).
- New `GET /admin/pairing-stats` endpoint: pick rate per (recipe, drink) pair, grouped by recipe tag.

---

## 3 · Research / nice-to-have

### 3.1 Wine SVD via approximate methods

**Why:** If demo users start rating wines explicitly and we see ≥5 explicit wine ratings per user, we'd want SVD personalization. Full Surprise SVD on a 21M-rating matrix is impractical — switch to `implicit` library or `LightFM` for matrix factorization on sparse feedback.

**What to add:**
- Extend `backend/ml/train_drink_cf.py` to produce a wine model via `implicit.AlternatingLeastSquares`.
- Update `serve_drink_cf.get_cf_scores` to dispatch wine candidates to it when warm.

**Caveat:** Likely unnecessary. The whole point of the v1 wine-side asymmetry (item-sim only, never SVD) is that demo users won't generate enough explicit wine ratings to need it.

### 3.2 Beta-style learning for drink scoring weights

**Why:** The recipe stack has `beta_updater.py` that learns each user's waste-aversion β from their behavior. We could do the same for drink scoring weights (e.g. learn that this user weights `expert_boost` more heavily because they pick rule-suggested drinks more often than CB-suggested ones).

**What to add:**
- New `backend/services/drink_beta_updater.py` mirroring `beta_updater.py`.
- New per-user fields on `User` (or a separate `UserDrinkPrefs` table) to store learned weights.
- Cron-style batch job invocation pattern.

**Caveat:** Adds operational complexity for unclear gain. The current static weights (0.45/0.25/0.20/0.10 for Path A) were tuned for the demo, not learned.

### 3.3 Replace hand-curated flavor lexicon with a learned mapping

**Why:** `flavor_bridge.py`'s ~50-entry lexicon is the most fragile part of v1. A learned mapping (e.g. logistic regression from recipe TF-IDF → beer style label, trained on whatever weak co-occurrence signal we can find) would scale better.

**What to add:**
- Training script that learns ingredient → flavor-token weights from review/recipe co-occurrence.
- Replace the lexicon dict with the learned model output, while keeping the same `bridge_ingredients` / `bridge_text` interface so nothing downstream changes.

**Caveat:** The hand-curated lexicon is interpretable and editable in 5 minutes. A learned model is none of those things. Only do this if expanding the lexicon becomes a real bottleneck (item 1.4 first).

### 3.4 Cross-domain CF via shared anchor users

**Why:** Genuine recipe ↔ beer / wine collaborative filtering would require users who appear in both Food.com and BeerAdvocate (or X-Wines). We have zero of these. But if someone built such a dataset, latent-space alignment becomes possible.

**What to add:** Out of scope for the foreseeable future. Listed here so the question "why didn't you do real cross-domain CF?" has a documented answer.

---

## Priority ordering (if you only have one afternoon)

1. **1.1** — train recipe CF artifacts (5 minutes, clears red tests)
2. **2.1** — upgrade X-Wines Slim (30 minutes including re-train, unlocks dormant code)
3. **1.4** — expand flavor lexicon (~1 hour, lifts CB recall meaningfully)
4. **2.5** — drink evaluation pipeline (~3 hours, gives you defensible numbers)

If you have a week, add **2.2** (pairing dataset) and **2.3** (MMR). The rest can wait for real usage signal.
