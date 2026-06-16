# HLD — Make the drinks module wine-only

## Problem

The drinks module was built to support multiple drink types (wine, beer, …).
Beer was already dropped, so the multi-type scaffolding is now dead weight before
opening the `drinks → main` PR:

- **DB**: joined-table inheritance `drinks` (parent) + `wines` (child) plus a
  `kind` discriminator column, and a `drink_events` table.
- **Code**: a `kind` query param accepting `'all'`/`'any'`/`'wine'`, a
  `_kind_to_filter` helper, and ~100 `.kind`/polymorphic/`kind_filter`/
  `cf_strategy`-routing references that only ever resolve to "wine".
- **Layout & prose**: `drinks/` folders, `drink_*` symbol names, and design docs
  that all describe a multi-drink system that no longer exists.

We want the module to read and behave as wine-only.

## Solution

On branch `wine-rename` (off `drinks`), in scope-increasing order as agreed:

1. **Schema collapse** — Merge `drinks` parent + `wines` child into a single flat
   `wines` table. Drop `kind` and all polymorphic inheritance. Rename
   `drink_events → wine_events` (`drink_id → wine_id`). ORM `Drink`/`Wine`/
   `DrinkEvent` → single `Wine` + `WineEvent`.
2. **Drop the `kind` machinery** — Remove `_kind_to_filter`, the `kind` query
   param, all `kind_filter` plumbing, per-kind CB splitting, `cf_strategy` kind
   routing. CB/CF become unconditionally wine.
3. **Backend folder renames** (cheap & safe) — `backend/db/drinks`,
   `backend/ml/drinks`, `backend/services/drinks`, `tests/drinks`, `data/drinks`
   → `.../wine`; collapse the redundant nested `wine/` sub-packages; rename the
   `test_drink_*.py` files to `test_wine_*.py`.
4. **Comments, docstrings & docs** — Update prose to wine-only, rename
   `docs/drink-recsys-*.md → wine-recsys-*.md`, fix now-stale file paths, and
   rename the `ENABLE_SYNTHETIC_DRINK_RATINGS` kill-switch constant.

**Deliberately kept** (API contract — would break the frontend): the `/drinks/*`
route paths, `/drink-events`, and the JSON field / schema-class names
(`drink_id`, `drink_name`, `DrinkScoreOut`, `DrinkEventIn`). The "Drinks For You"
product label is also kept.

Existing seeded data is dev-only and disposable: **reset & reseed** rather than
migrate. `reset_drinks.py → reset_wines.py` recreates `wines` + `wine_events`,
then the existing wine seed repopulates.

## Flow

```
Seed:   reset_wines  →  drop wines/wine_events  →  recreate  →  seed wines
Read:   GET /drinks/ranked            (Path B)    — no kind param
        GET /drinks/pairings/{rid}    (Path A, + expert boost)
        GET /drinks/search, /drinks/{id}
Rank:   candidates (popularity) → CB + CF + expert + prior → blend → top_n
Write:  POST /drink-events  →  WineEvent row
Side:   high recipe rating  →  synthesizer  →  synthetic WineEvent
```

## Sketch

```
# models.py — one table, no inheritance, no kind
class Wine(Base):
    table "wines"
    id, name, producer, country, style, abv,
    avg_rating, n_ratings, harmonize_csv, review_tokens_csv,   # were on Drink
    grapes_csv, body, acidity, region                          # were on Wine child

class WineEvent(Base):
    table "wine_events"
    id, user_id, wine_id -> wines.id, event_type, rating, synthetic, created_at

# routers/drinks.py — kind param gone, query Wine directly (routes kept)
GET /drinks/ranked(user_id, top_n):
    candidates = top-N wines by bayesian popularity
    cb = cb_for_user(user_id)            # no kind_filter
    cf = get_cf_scores(user_id, [w.id])  # ids only
    return rank_wines_for_user(...)

# serve_cf / serve_cb / scoring — drop kind args & cf_strategy routing
cf_strategy_name(n_explicit):            # was (n_explicit, kind)

# reset_wines.py
WINE_TABLES = [WineEvent.__table__, Wine.__table__]   # was 3 incl. Drink
```

## Files added and changed

**Renamed (paths)**
- `backend/db/reset_drinks.py → backend/db/reset_wines.py`
- `backend/db/drinks/ → backend/db/wine/`
- `backend/ml/drinks/ → backend/ml/wine/` (+ collapse `training/wine/` up one level)
- `backend/services/drinks/ → backend/services/wine/`
- `tests/drinks/ → tests/wine/` (+ `test_drink_*.py → test_wine_*.py`)
- `data/drinks/ → data/wine/` (flattened the redundant `wine/` nesting)
- `docs/drink-recsys-{design,steps,future}.md → wine-recsys-*.md`

**Edited (content)**
- `backend/db/models.py` — single `Wine` + `WineEvent`; drop `kind`/polymorphic.
- `backend/routers/drinks.py` — query `Wine`; drop `_kind_to_filter`/`kind`
  param; `WineEvent`. (Routes + response field names kept.)
- `backend/services/wine/{scoring,synthesizer,expert_pairing}.py` — `Wine`/
  `WineEvent`; drop kind; rename kill switch.
- `backend/ml/wine/serving/{serve_cf,serve_cb,flavor_bridge,cold_start}.py` —
  drop `kind`/`kind_filter` args, ids-only CF.
- `backend/ml/wine/training/{train_cb,item_similarity,*wine*}.py` — `Wine`/
  `WineEvent`; fixed `parents[N]` offsets after the folder collapse.
- `backend/routers/recipes.py`, `backend/main.py` — updated import paths.
- `tests/wine/*` — drop `kind`/`'all'` cases; `WineEvent`; new import paths.
- `README.md`, `docs/*.md` — paths, schema symbols, cross-doc links.

## New dependencies

None. Schema simplification + renames + dead-code removal only.

## Alternative solutions

**A. Rename only, keep 2 tables + inheritance** (rejected) — keeps the exact
polymorphic scaffolding we're deleting; `kind` lingers with one value.

**B. Also rename routes + frontend (`/drinks → /wines`)** (deferred) — zero
"drink" vocabulary, but it's an API-contract break touching the frontend with no
functional gain. Left as a possible later cosmetic PR.

**C. Data migration instead of reset** (rejected) — dev-only seeded data is
reproducible from the wine seed in seconds; a migration is pure overhead.

## Why this solution

The schema collapse + kind removal is where the real value is: it deletes
functional complexity (inheritance, discriminator, kind routing). The folder /
prose renames make the module *read* as wine-only at near-zero risk (pure import
churn, all behind passing tests). Keeping the `/drinks/*` routes and JSON field
names is the one deliberate line we don't cross, because that's the only part
that would break the frontend — so it stays for a separate, frontend-aware PR.
Reset-and-reseed is justified by disposable dev data.
