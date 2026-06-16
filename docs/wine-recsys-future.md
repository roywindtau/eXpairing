# Wine Recommender — Roadmap

## Current state (as of 2026-05-30)

- X-Wines Full (100K wines, 21M ratings) downloaded, cleaned, seeded into DB
- DB schema uses joined table inheritance (Drink → Wine)
- Wine CF model trained (confidence-weighted ALS — see `docs/wine-cf-experiments.md`)

---

## Product features

Two scenarios:

**Scenario 1 — Pair a wine to a recipe**
- `expert(wine, recipe)` + `taste(user, wine)` + `curiosity`

**Scenario 2 — Recommend me a wine**
- `taste(user, wine)` + `curiosity`

**Signal definitions:**
- `taste(user, wine)` — CF model trained on wine ratings (ALS)
- `expert(wine, recipe)` — Harmonize rules + sommelier heuristics (fatty→tannin, spicy→low acidity, etc.) + CB model
- `curiosity` — serve-time diversity parameter (MMR or epsilon), not trained

---

## User stories

### US-1: Train wine CF — DONE

Confidence-weighted ALS trained on wine ratings (chosen over Funk SVD on
ranking metrics — see `docs/wine-cf-experiments.md`). Covers Scenario 2
(recommend me a wine) for warm users. Item-similarity matrix built for the
cold-start / sparse serving path.

### US-2: Add exploration

Wire curiosity parameter into the serving layer.

- Implement MMR reranking in wine scoring
- Tunable λ parameter per request
- Mirror recipe stack's MMR implementation

→ After US-1 + US-2: **Scenario 2 fully covered.**

### US-3: Sommelier CB — pairing model

Train content-based models for Scenario 1 (pair a wine to a recipe). Two CB
signals are still needed:

- **Wine→food CB** — match a wine to a recipe via the flavor bridge
- **Wine→wine CB** — surface wines similar to ones the user has liked

Plus:
- Define sommelier rules: fatty→high tannin, spicy→low acidity, rich→full body, etc.
- Extend Harmonize matching with these rules in `expert_pairing.py`
- Train CB model on wine attributes + recipe features via flavor bridge
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
