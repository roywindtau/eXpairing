# Fridge2Fork

A recipe recommender system that ranks recipes to minimize food waste,
personalized to what's expiring in your fridge and how you cook.

Built for the Recommender Systems workshop at Tel Aviv University.

---

## What it does

- Scan your fridge or add items manually with expiry dates
- Ingredient autocomplete in the pantry form — suggestions drawn from the Food.com corpus as you type
- Get a ranked feed of recipes weighted by:
  - **Collaborative filtering** — community co-rating patterns via item-based CF (cold start) or matrix factorization (warm); CF has the highest single weight
  - **Expiry urgency** — recipes that use your soon-to-expire items score highest; urgency is normalized by pantry size (not recipe length) so a complex recipe using *all* your expiring items beats a simple recipe using only one
  - **Ingredient match** — penalizes recipes needing many extra purchases (per-user β)
  - **Content-based similarity** — pantry/taste TF-IDF profile matching (warm users get a taste-profile CB built from rated recipes)
- **Score calibration** — each component is min-max normalized across all candidates before blending, so no single dimension dominates due to scale differences
- **MMR diversity reranking** — feed is re-ranked with Maximal Marginal Relevance (λ=0.7) to reduce ingredient monotony; the highest-scoring recipe is always first
- **Sort the feed** by any score component (CF score, CB score, expiry urgency, pantry match) — client-side within the 20 loaded recipes
- **Skip exclusion** — recipes dismissed in the last 7 days are hidden from the feed
- Cold start: new users get personalized recommendations immediately via diet tags + pantry-seeded item-based CF; fallback preference scores ensure CF is never zero
- **Implicit feedback augmentation** — cook events converted to synthetic ratings (`max(3.0, 4.0 − min(n_missing, 3) × 0.3)`) and merged into matrix factorization training data when explicit ratings are absent
- β learning: waste-aversion preference drifts automatically toward revealed behavior; Profile page warns when stated vs. revealed β diverge by > 10%
- Recipe detail page: full ingredients list + numbered step-by-step instructions from Food.com data
- **Shopping list**: add missing ingredients from any recipe to a persistent buy-list, check them off while shopping, clear purchased items

---

## Architecture

```
frontend/          React + TypeScript (Vite)
backend/
  main.py          FastAPI app
  routers/         pantry, recipes, users, vision, shopping, drinks
  services/        scoring, expiry, ingredient_match, beta_updater, vision_agent,
                   drink_scoring, drink_synthesizer, expert_pairing
  ml/              recipes:  train_cf, train_cb, item_similarity, serve_cf, serve_cb, cold_start
                   drinks:   train_drink_cb, train_drink_cf, drink_item_similarity,
                             serve_drink_cb, serve_drink_cf, drink_cold_start, flavor_bridge
  db/              SQLAlchemy models (User, Recipe, Drink, UserEvent, DrinkEvent, …) + seed scripts
data/              download scripts (Food.com + drinks), EDA notebook
train_pipeline.sh  one-shot training pipeline (recipes + drinks)
tests/             530+ backend tests (unit + behavioral integration) + 63 E2E
```

## Drink recommender (beer + wine)

A second recommender module pairs **drinks** to the user, in two contexts:

- **Path A — Pair with a recipe.** On any recipe detail page, a "Pair this with…" panel shows 4–6 ranked drinks for that specific dish. Surfaces an expert-rules boost (X-Wines Harmonize match for wines, hand-coded style heuristics for beers).
- **Path B — Drinks For You.** A standalone `/drinks` page ranks drinks by the user's food + drink history. Uses the same CB/CF machinery as recipes, plus a flavor-bridge lexicon that maps recipe ingredients → drink-compatible flavor tokens.

Both paths share four signal sources blended via min-max calibrated weighted sum:
- **CB** — TF-IDF over drink descriptors (style/variety/wine type/grapes/Harmonize) bridged from the recipe or user-history side
- **CF** — Bayesian-smoothed popularity → item-item cosine (cold-start / wines) → Surprise SVD (warm beers, ≥5 explicit ratings)
- **Expert rules** — Path A only; rule-based pairing knowledge
- **Popularity prior** — `avg_rating · log1p(n_ratings)` tiebreaker

Cold-start solution: when a user rates a recipe ≥ 4.0, the **drink synthesizer** infers compatible drinks via CB+expert and inserts `DrinkEvent` rows with `synthetic=True, rating=4.0`. These seed the item-similarity path until the user accumulates real drink ratings.

Both UI surfaces translate the dominant signal into a plain-English **"why this drink"** line (`🎯 Harmonizes with Beef, Lamb, Grilled` for expert hits, `🍽️ Matches your food taste` for CB-driven picks, etc.) — the raw algorithm name never leaks to the user.

### Drink-recommender docs

- [`docs/drink-recsys-design.md`](docs/drink-recsys-design.md) — architecture, scoring formulas, schema, design rationale, suggested reading order
- [`docs/drink-recsys-steps.md`](docs/drink-recsys-steps.md) — chronological as-shipped record of the 11 build steps
- [`docs/drink-recsys-future.md`](docs/drink-recsys-future.md) — deferred work, prioritized by impact-per-effort

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/drinks/ranked?user_id=&kind=&top_n=` | Path B "Drinks For You" |
| `GET`  | `/drinks/pairings/{recipe_id}?user_id=&kind=&top_n=` | Path A pairing panel |
| `GET`  | `/drinks/search?q=&kind=&limit=` | Browse / search |
| `GET`  | `/drinks/{drink_id}` | Drink detail |
| `POST` | `/drink-events` | Log a drink rating (1–5 stars) |

`kind` accepts `beer`, `wine`, or `all` (default). The synthesizer fires only on **recipe** rate events; drink rate events feed directly into SVD / item-sim.

### Datasets

- **[Beer Reviews](https://www.kaggle.com/datasets/rdoume/beerreviews)** (Kaggle) — ~66k beers, ~1.5M ratings
- **[X-Wines](https://github.com/rogerioxavier/X-Wines)** (Test slice from GitHub) — 100 wines, ~1k ratings

Download both with:
```bash
python3 -m data.drinks.download_beer
```

### Train + try the demo

```bash
# Build the drink models (~3 minutes on the Test slice)
./train_pipeline.sh --drinks-only

# Restart the backend so it picks up the new artifacts
uvicorn backend.main:app --reload

# In the frontend, click "Drinks" in the nav (Path B)
# Or open any recipe → scroll to "Pair this with…" panel (Path A)
```

You can also probe the API directly:

```bash
# Path B — drinks for user 1
curl "http://localhost:8000/drinks/ranked?user_id=1&kind=wine&top_n=5" | jq

# Path A — wine pairings for recipe 42
curl "http://localhost:8000/drinks/pairings/42?user_id=1&kind=wine&top_n=6" | jq

# Rate a drink
curl -X POST "http://localhost:8000/drink-events" \
     -H "Content-Type: application/json" \
     -d '{"user_id":1,"drink_id":1,"event_type":"rate","rating":4.5}'
```

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.9 or later | `python3 --version` |
| pip | any | `pip3 --version` |
| Node.js | 18 or later | `node --version` |
| npm | 9 or later | `npm --version` |

> **macOS note:** System Python 3.9 works. All commands use `python3`. If you have a newer Python via pyenv/conda/Homebrew, use that instead.

---

## Quick start (local, no ML models needed)

> All commands run from the **project root** unless stated otherwise.

### Step 1 — Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### Step 2 — Seed the demo database

Creates `fridge2fork.db` with 20 diverse recipes (varied dietary tags, ratings, steps) and a demo pantry:

```bash
python3 -m backend.db.seed_dev
```

Expected output:
```
Seeded 20 recipes.
Created user id=1 with 10 pantry items.
```

### Step 3 — Start the backend

```bash
python3 -m uvicorn backend.main:app --reload --port 8000
```

Verify it's running:
- **Health check:** http://localhost:8000/health → `{"status":"ok"}`
- **Interactive API docs:** http://localhost:8000/docs

### Step 4 — Install frontend dependencies (once)

```bash
cd frontend
npm install
```

### Step 5 — Start the frontend

```bash
npm run dev
```

Open **http://localhost:5173** — you'll see the onboarding screen.

---

## Running tests

### Unit + behavioral integration tests

```bash
# from project root
python3 -m pytest tests/ -v
```

Expected: **530+ tests** pass in ~5–10 seconds (most are pure unit; live-backend behavioral tests require the backend running at localhost:8000).

Run only unit tests (no backend needed):

```bash
python3 -m pytest tests/ -v --ignore=tests/test_ml_behavior.py
```

### End-to-end Playwright tests

```bash
cd frontend
npx playwright test --reporter=list
# 63 tests, ~45 seconds (requires both backend and frontend running)
```

---

## Full ML training (Food.com dataset)

The quick start uses 20 hand-crafted dev recipes. For real CF and CB models trained on 230k+ recipes:

### Get the dataset

1. Create a [Kaggle](https://www.kaggle.com) account
2. **Account → Create New Token** — downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json`
4. Download:

```bash
python3 -m data.download_foodcom
```

### Run the full pipeline

```bash
chmod +x train_pipeline.sh
./train_pipeline.sh                  # full pipeline: recipes + drinks (~20 min)
./train_pipeline.sh --skip-drinks    # recipes only (original behavior, ~15 min)
./train_pipeline.sh --drinks-only    # drinks only (~3 min on the X-Wines Test slice)
./train_pipeline.sh 10000            # 10k recipes for quick dev (still does drinks)
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `seed_recipes.py` | 231k recipes in DB |
| 2 | `seed_ratings.py` | 1.1M ratings in DB |
| 3 | `item_similarity.py` | `models/item_sim_matrix.npz` (sparse, top-50 per recipe) |
| 4 | `train_cf.py` | `models/cf_model.pkl` (biased MF warm CF) |
| 5 | `train_cb.py` | `models/cb_matrix.npz` (TF-IDF content-based) |
| 6 | `evaluate.py` | `models/eval_results.json` (RMSE, Precision@K, NDCG@K) |

After training, restart the backend — it picks up model files automatically.

---

## Docker

```bash
docker-compose up --build
# backend  → http://localhost:8000
# frontend → http://localhost:5173
```

---

## Optional: GPT-4o vision scanning

```bash
export OPENAI_API_KEY=sk-...
python3 -m uvicorn backend.main:app --reload --port 8000
```

Without it, the **Demo Scan** button still works (returns realistic fake data).

---

## ML components

### Scoring formula

```
final_score = γ · cf_score          (CF base — highest single weight)
            + δ · cb_score          (CB ingredient profile boost)
            + α · expiry_urgency    (domain: waste minimization)
            + β · match_ratio       (domain: ingredient availability, per-user)
```

Default weights: γ=0.35, α=0.35, β=0.20 (per-user, learned), δ=0.10.
When CF or CB models are unavailable, their weights redistribute to α and β.

**Score calibration:** Each component is min-max normalized across the full candidate pool before blending. This prevents a wide-ranging expiry signal from drowning out a narrow CF signal on sparse data.

**MMR reranking:** The top 60 candidates (3 × top_n) pass through Maximal Marginal Relevance selection with λ=0.7. The output feed has ingredient diversity; the highest-scored recipe is always first.

**Skip exclusion:** Recipes dismissed via "Skip" are excluded from the candidate pool for 7 days.

### Collaborative filtering — two strategies

**Cold start** (< 5 ratings, or no trained model):
- Selects seed recipes matching the user's diet tags + pantry ingredients
- Scores via tag/pantry overlap (preference scores); uses item-item cosine similarity when model available
- Seeds diversified across cuisines — no echo chambers
- CF scores are always non-zero (preference-score fallback when no model files)
- Works from first page load, no rating history required

**Warm user** (≥ 5 ratings, model present):
- Biased matrix factorization trained on Food.com ratings
- Formula: `predicted(u,r) = μ + b_u + b_r + p_u · q_r^T` (learned by SGD)
- Predicts per-user, per-recipe rating from learned latent vectors
- Transition is automatic at exactly 5 ratings
- Implemented via Surprise's `SVD` class (which is biased MF, not true SVD)

### Item-item similarity matrix

Built from the 230k×196k user-rating matrix using fully sparse operations:
- Filters to recipes with ≥5 ratings (reduces to ~51k)
- Mean-centered per user; L2-normalized
- Cosine similarity computed in chunks — avoids materializing the full matrix
- Result: ~51k×51k sparse matrix, top-50 neighbors per recipe

### Content-based similarity

TF-IDF vectors over ingredient tokens (unigrams + bigrams, 20k vocab).

- **Cold-start users** (< 5 ratings): cosine similarity between pantry vector and each recipe vector captures cuisine affinity (miso + soy → Japanese recipes)
- **Warm users** (≥ 5 ratings): taste-profile CB — weighted average of rated recipe TF-IDF vectors, weight = (rating − 3.0); captures explicit preference beyond pantry contents

### Implicit feedback augmentation (train_cf.py)

Cook events augment matrix factorization training with synthetic ratings when no explicit star rating exists:

```
implicit_rating = max(3.0, 4.0 − min(n_missing, 3) × 0.3)
  n_missing = 0 → 4.0   (had everything: strong positive)
  n_missing = 1 → 3.7
  n_missing = 2 → 3.4
  n_missing = 3 → 3.1   (weak but still positive)
```

Explicit ratings always take precedence. The net effect: faster warm-up for users who cook frequently but rate rarely.

### β learning (beta_updater.py)

Daily batch. Compares stated vs revealed waste-aversion from cook events.
EMA: `new_β = (1−lr)·current_β + lr·revealed_β`, lr=0.15.
The Profile page displays an amber warning when stated β diverges from revealed β by > 0.1.

### Offline evaluation (evaluate.py)

| Metric | Description |
|--------|-------------|
| RMSE / MAE | Rating prediction accuracy vs. held-out set |
| Precision@K / Recall@K | Top-K ranking quality |
| **NDCG@K** | Graded ranking quality (rewards rank position of highly-rated items) |
| Ablation | CF / CB / domain-only vs. full hybrid comparison |
| **Lifecycle simulation** | NDCG@10 vs. n_ratings — validates soft CF blend ramps smoothly |
| **Weight grid search** | Grid over (γ, α) — validates/updates DEFAULT_GAMMA, DEFAULT_ALPHA |

```bash
python3 -m backend.ml.evaluate              # full suite
python3 -m backend.ml.evaluate --lifecycle  # cold→warm ramp
python3 -m backend.ml.evaluate --tune       # weight grid search
```

---

## Project structure

```
backend/
  main.py                    FastAPI entry point
  routers/
    pantry.py                GET/POST/DELETE pantry items; GET /pantry/suggest (autocomplete)
    recipes.py               GET /recipes/ranked, /recipes/search,
                               GET /recipes/{id}, POST /events
    users.py                 GET/PUT user profile + stats
    vision.py                GET /vision/mock, POST /vision/scan,
                               POST /vision/confirm/{user_id}
    shopping.py              GET/POST/PATCH/DELETE /shopping/{user_id}
    drinks.py                GET /drinks/ranked, /drinks/pairings/{rid},
                               /drinks/search, /drinks/{id}; POST /drink-events
  services/
    scoring.py               Core ranking formula (RecipeScore dataclass)
    expiry.py                Urgency score (exponential decay)
    ingredient_match.py      Fuzzy ingredient overlap
    beta_updater.py          Daily preference learning job
    vision_agent.py          GPT-4o vision + ingredient canonicalization
    drink_scoring.py         Drink ranking — Path A + Path B formulas
    drink_synthesizer.py     Recipe-rate ≥ 4.0 → synthetic DrinkEvents (cold start)
    expert_pairing.py        Rule-based expert boost (Harmonize + beer style heuristics)
  ml/
    cold_start.py            Preference-seeded cold start CF (with fallback)
    item_similarity.py       Sparse item-item similarity matrix (training)
    train_cf.py              Biased MF training (scikit-surprise)
    train_cb.py              TF-IDF training (sklearn)
    serve_cf.py              CF serving — warm/cold auto-selection
    serve_cb.py              CB serving — cosine similarity at request time
    user_vector.py           Pantry → TF-IDF vector utility
    evaluate.py              RMSE, Precision@K, Recall@K, ablation
    flavor_bridge.py         Ingredient → drink-side flavor token lexicon
    train_drink_cb.py        Drink TF-IDF training (style + variety + Harmonize)
    train_drink_cf.py        Drink SVD training (beers only — wines too sparse)
    drink_item_similarity.py Drink item-item cosine matrices (one per kind)
    drink_cold_start.py      Bayesian popularity + item-sim seed scores
    serve_drink_cb.py        Drink CB serving — cb_for_recipe + cb_for_user
    serve_drink_cf.py        Drink CF serving — routes by warmth + kind
  db/
    models.py                SQLAlchemy ORM (User, PantryItem, Recipe, UserEvent,
                               ShoppingListItem, Drink, DrinkEvent)
    database.py              Engine, session, init_db
    seed_dev.py              Dev seed: 20 recipes with tags/steps + demo pantry
    seed_recipes.py          Load Food.com CSV → Recipe table
    seed_ratings.py          Load Food.com ratings → UserEvent table
  drinks/
    seed_drinks.py           Load Beer Reviews → Drink table (wine pending)
    seed_ratings.py          Load beer ratings → DrinkEvent (external users)
  canonicalizer/
    ingredient_map.py        Rule-based + fuzzy product name cleaner
    openfoodfacts.py         Barcode/name lookup via OFF API

frontend/src/
  App.tsx                    Router, auth guard, stale-user detection, nav
  api/client.ts              Axios client + all TypeScript types
  hooks/useUserId.ts         Persists user ID in localStorage
  index.css                  CSS custom properties + base styles
  api/drinks.ts              Typed wrappers for the 5 drink endpoints
  components/
    ExpiryBadge.tsx          Color-coded days-remaining badge + urgency bar
    IngredientAutocomplete.tsx  Debounced autocomplete input backed by /pantry/suggest
    RecipeCard.tsx           Score ring + match ring, explainer, Cook→Rate flow, Buy missing button
    ScoreExplainer.tsx       4-component score breakdown bars (unavailable = grayed)
    VisionScanner.tsx        Photo scan → confirm → add to pantry
    DrinkCard.tsx            Path-B drink card with kind badge, breakdown, star rating
    DrinkPairingPanel.tsx    Path-A pairing panel on RecipeDetailPage ("Pair this with…")
  pages/
    OnboardingPage.tsx       First-run: name, beta slider, diet tags
    PantryPage.tsx           Pantry management with expiry rows + scan button
    RecipeFeedPage.tsx       Ranked recipe feed with CF strategy banner + sort-by dropdown
    RecipeDetailPage.tsx     Full recipe + drink pairing panel at bottom
    BrowsePage.tsx           Search/filter all recipes (clickable → detail)
    DrinksForYouPage.tsx     Path-B drinks feed (kind toggle, sort, CF banner)
    ProfilePage.tsx          Beta + diet tags + CF progress bar
    ShoppingListPage.tsx     Buy-list: check off items, clear purchased, source recipe attribution

tests/
  test_scoring.py            Unit: expiry decay, fuzzy match, ranking, weights
  test_beta_updater.py       Unit: beta math, DB integration, convergence
  test_cf.py                 Unit: MF routing, item-sim, warm/cold threshold
  test_cold_start.py         Unit: seed selection, diversification, fallback scores
  test_vision_agent.py       Unit: brand stripping, fuzzy match, mock scan
  test_evaluate.py           Unit: RMSE math, Precision@K logic
  test_shopping.py           Unit: shopping list CRUD + deduplication (20 tests)
  test_improvements.py       Unit: cook augmentation, calibration, MMR, CB taste
                               profile, revealed β, NDCG, skip exclusion (46 tests)
  test_ml_behavior.py        Integration: live-API behavioral tests (38 tests)
                               — CF cold/warm, pantry effect, beta, vision

frontend/e2e/
  fridge2fork.spec.ts        Playwright E2E: 63 tests across all pages
  demo.spec.ts               Full feature demo recording script

data/
  download_foodcom.py        Kaggle download (Food.com)
  download_drinks.py         Kaggle (Beer Reviews) + GitHub raw (X-Wines Test)
  explore_foodcom.ipynb      EDA notebook

train_pipeline.sh            One-shot training pipeline (recipes + drinks,
                               supports --skip-drinks / --drinks-only)

docs/
  drink-recsys-design.md     Drink recommender architecture + rationale
  drink-recsys-steps.md      11-step implementation plan with acceptance criteria

models/                      Trained artifacts (git-ignored)
  # recipes
  cf_model.pkl               Biased MF model (scikit-surprise SVD class)
  item_sim_matrix.npz        Sparse item-item similarity (top-50 per recipe)
  item_sim_recipe_ids.npy    Recipe ID index for similarity matrix
  cb_matrix.npz              TF-IDF recipe embeddings
  cb_vectorizer.pkl          Fitted TfidfVectorizer
  eval_results.json          Offline evaluation results
  # drinks
  drink_cb_matrix.npz        TF-IDF drink embeddings
  drink_cb_vectorizer.pkl    Fitted TfidfVectorizer for drinks
  drink_cb_ids.npy           Drink ID index (aligned with matrix rows)
  drink_cb_kinds.npy         Per-row kind (beer/wine) for kind-filtered queries
  drink_cb_meta.json         Trained-on counts, hyperparams
  drink_cf_model.pkl         Surprise SVD trained on non-synthetic beer ratings
  drink_cf_meta.json         SVD hyperparams + filtered rating counts
  drink_sim_beer.npz         Beer item-item cosine similarity (≥5 ratings)
  drink_sim_beer_ids.npy     Beer ID index for the beer sim matrix
  drink_sim_wine.npz         Wine item-item cosine similarity (≥2 ratings)
  drink_sim_wine_ids.npy     Wine ID index for the wine sim matrix
  drink_sim_meta.json        Filter thresholds + per-kind counts
```

---

## Team

TBD
---

## Dataset

[Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
— 231k recipes, 1.1M ratings, rich ingredient and tag features.
