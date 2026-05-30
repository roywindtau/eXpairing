# HLD: Drinks Schema — Unified vs Split Tables

## Problem

The current `drinks` table uses a single-table design with a `kind` discriminator (`"beer"` | `"wine"`). Beers and wines have different attributes, and the product requires modular queries — "recommend me a wine", "recommend me a beer", "pair me anything." The question is whether the schema should reflect that modularity.

## Solution

Migrate to **joined table inheritance**: a shared `drinks` base table for common columns, with `beers` and `wines` child tables for kind-specific attributes. SQLAlchemy supports this natively.

## Flow

```
drinks (base)
  id, name, abv, avg_rating, n_ratings, country, producer, kind
       ↑                          ↑
beers (child)               wines (child)
  style, ibu,               wine_type, grapes_csv,
  avg_aroma, avg_taste,     harmonize_csv, body,
  avg_palate, avg_appearance  acidity
```

- Query all drinks → query `drinks` base table
- Query wines only → join `drinks` + `wines`
- Query beers only → join `drinks` + `beers`
- Pair anything for a recipe → query base table, filter by `Harmonize`/style

## Sketch

```
class Drink(Base):
    id, name, abv, avg_rating, n_ratings, country, producer
    kind  # discriminator

class Beer(Drink):
    style, ibu, avg_aroma, avg_taste, avg_palate, avg_appearance

class Wine(Drink):
    wine_type, grapes_csv, harmonize_csv, body, acidity
```

Seeder:
```
for row in clean_wines.csv:
    insert Wine(id=WineID, name=WineName, ...)

for row in beer_reviews.csv:
    insert Beer(id=beer_id, name=beer_name, ...)
```

Serving (modular):
```
recommend(user, kind="wine")  → query Wine joined Drink
recommend(user, kind="beer")  → query Beer joined Drink
recommend(user, kind=None)    → query Drink base table
```

## Files Added

- `backend/db/models.py` — redefine `Drink`, add `Beer`, `Wine` with joined inheritance
- `backend/db/drinks/seed_wines.py` — new wine seeder using `Wine` model
- `backend/db/migrations/001_split_drinks.py` — migration to split existing `drinks` table

## Files Changed

- `backend/db/drinks/seed_drinks.py` — update to use `Beer` model instead of `Drink`
- `backend/routers/drinks.py` — update queries to use correct model per kind
- `backend/services/drinks/scoring.py` — update drink fetching
- `backend/services/drinks/synthesizer.py` — update candidate queries
- `backend/ml/drinks/serving/serve_cb.py` — update drink fetching
- `backend/ml/drinks/serving/serve_cf.py` — update drink fetching

## New Dependencies

None — SQLAlchemy joined table inheritance is built-in.

## Alternative Solutions

### A. Keep unified table (status quo)
Single `drinks` table, nullable columns per kind, `kind` discriminator.
- ✅ Zero migration, all existing code works today
- ✅ "Pair anything" is one clean query
- ❌ Wide table with many nulls
- ❌ Adding wine-specific features feels wrong in a shared model

### B. Concrete table inheritance
Completely separate `beers` and `wines` tables, no shared base table.
- ✅ Each table is fully independent, cleanest schema
- ❌ "Pair anything" requires UNION — awkward in SQLAlchemy
- ❌ Shared columns (avg_rating, n_ratings) duplicated in both tables
- ❌ Most code needs to handle two separate models everywhere

### C. Joined table inheritance (chosen)
Shared base + child tables per kind.
- ✅ Modular — each kind has its own model and clean schema
- ✅ "Pair anything" still works on base table
- ✅ Shared columns live in one place
- ❌ Migration required, several files need updating
- ❌ Slightly more complex ORM queries (SQLAlchemy handles it, but joins add overhead)

## Why This Solution

The five product queries (wine rec, beer rec, food-based rec, sommelier pairing, user-based pairing) split 2/5 kind-specific and 3/5 cross-kind. Concrete table inheritance (Option B) makes the 3 cross-kind queries awkward. The unified table (Option A) makes the 2 kind-specific queries feel wrong at the model level and blocks clean modularity. Joined inheritance (Option C) handles both cleanly — kind-specific queries get a typed model, cross-kind queries use the base table. The migration cost is real but one-time.
