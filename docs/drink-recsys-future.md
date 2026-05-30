# Drink Recommender — Roadmap

## Current state (as of 2026-05-30)

- X-Wines Full (100K wines, 21M ratings) downloaded, cleaned, seeded into DB
- BeerAdvocate (~66K beers, 1.58M reviews) cleaned, seeded into DB
- DB schema migrated to joined table inheritance (Drink → Beer / Wine)
- No models trained yet — models/ is empty

---

## Product features

Two scenarios, six permutations:

**Scenario 1 — Pair a drink to a recipe**
- Wine pairing: `expert(wine, recipe)` + `taste(user, wine)` + `curiosity`
- Beer pairing: `expert(beer, recipe)` + `taste(user, beer)` + `curiosity`
- Any drink: both, weighted by `preference(user, wine vs beer)`

**Scenario 2 — Recommend me a drink**
- Wine: `taste(user, wine)` + `curiosity`
- Beer: `taste(user, beer)` + `curiosity`
- Any drink: both, weighted by `preference(user, wine vs beer)`

**Signal definitions:**
- `taste(user, wine)` — CF model trained on wine ratings CSV
- `taste(user, beer)` — CF model trained on beer ratings CSV
- `expert(drink, recipe)` — Harmonize rules + sommelier heuristics (fatty→tannin, spicy→low acidity, etc.) + CB model
- `preference(user, wine vs beer)` — ratio of wine/beer events in DrinkEvent at serve time
- `curiosity` — serve-time diversity parameter (MMR or epsilon), not trained

---

## User stories

### US-1: Train CF (beer + wine) — IN PROGRESS (pre-train-cf branch)

Train Funk SVD on each drink kind's ratings CSV directly (not from DB).
Covers Scenario 2 (recommend me a drink) for warm users.

- Rewrite `train_cf.py` to load from CSV instead of DrinkEvent table
- Train beer CF on `data/drinks/beer_reviews.csv`
- Train wine CF on `data/drinks/clean_ratings.csv`
- Build beer + wine item-similarity matrices (cold start)

### US-2: Add exploration

Wire curiosity parameter into the serving layer.
Covers the diversity requirement across all 6 permutations.

- Implement MMR reranking in drink scoring
- Tunable λ parameter per request
- Mirror recipe stack's MMR implementation

→ After US-1 + US-2: **Scenario 2 fully covered.**

### US-3: Sommelier CB — pairing model

Train content-based model for Scenario 1 (pair a drink to a recipe).

- Define sommelier rules: fatty→high tannin, spicy→low acidity, rich→full body, etc.
- Extend Harmonize matching with these rules in `expert_pairing.py`
- Train CB model on wine/beer attributes + recipe features via flavor bridge
- Blend: `pairing_score = α × rules + (1-α) × cb_score`

→ After US-3: **Scenario 1 fully covered.**

---

## Deferred / nice-to-have

### Offline evaluation pipeline
Drinks have no equivalent of `backend/ml/evaluate.py`. Need RMSE, Precision@K,
Recall@K, NDCG@K on a held-out set before we can measure model quality.
Build `backend/ml/evaluate_drinks.py` after US-1.

### Multi-event types
Only `event_type="rate"` supported. Add `like`, `skip`, `save` and convert
to implicit ratings at training time (like → 4.0, save → 3.5, skip → 1.5).

### `paired_recipe_id` analytics
Record which recipe triggered a drink pairing event so we can measure
pick rates per (recipe, drink) pair.

### MMR for drinks
Mirror recipe stack's MMR diversity reranking in drink scoring.
Already planned as part of US-2 above.

### Expand flavor lexicon
`flavor_bridge.py` has ~50 entries — most ingredients on the full Food.com
corpus produce empty bridge docs. Add top-50 missing ingredients by frequency.
