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
  routers/         pantry, recipes, users, vision, shopping, wine
  services/        scoring, expiry, ingredient_match, beta_updater, vision_agent,
                   wine:     scoring, synthesizer, expert_pairing
  ml/              recipes:  train_cf, train_cb, item_similarity, serve_cf, serve_cb, cold_start
                   wine:     train_cb, train_wine_als, item_similarity,
                             serve_cb, serve_cf, cold_start, flavor_bridge
  db/              SQLAlchemy models (User, Recipe, Wine, UserEvent, WineEvent, …) + seed scripts
                   wine: seed_wines, reset_wines, compute_wine_stats
data/              download scripts (Food.com + wine), EDA notebook
tests/             backend tests (unit + behavioral integration) + Playwright E2E
```

## Wine recommender

A second module recommends **wines**. The end goal is two features, each a
blend of collaborative filtering and content-based scoring:

- **Recommend me a wine** = wine-CF + wine↔wine CB
- **Pair with a recipe** = wine-CF + wine↔food CB

Three models back these: **wine-CF** (trained — confidence-weighted ALS,
`models/wine_als_*`), **wine↔wine CB** (built — structured weighted vector), and
**wine↔food CB** (to do, for recipe pairing).

> **Current state.** **"Recommend me a wine" is personalized** — a blend of
> collaborative filtering (ALS) and content-based scoring with a popularity cold
> start. The **pair-with-recipe** feature (wine↔food CB) is still future work.
>
> **How the ranking works** (`backend/services/wine/scoring.py`):
> - **Cold start** (0 ratings) → top wines by Bayesian-smoothed popularity.
> - **Warming** (1–4 ratings) → content-based taste profile + popularity.
> - **Warm** (≥5 ratings) → `0.5·CF + 0.5·CB`, min-max calibrated.
> - **Style filter** — candidates restricted to the styles the user drinks (or
>   an explicit style choice from the UI).
>
> The **content-based** model is a *structured weighted vector* (not TF-IDF, not
> embeddings — X-Wines has no free text): grape (multi-hot), region (rolled up
> from 2,160 appellations to ~107 parents), acidity/body (ordinal), abv
> (normalized), compared by cosine. Per-field weights come from a sommelier's
> palate-first prior (acidity + body dominate). See
> [docs/wine-cb-branch-summary.md](docs/wine-cb-branch-summary.md).

On the `/wine` page the user clicks **"Suggest me a wine"**, optionally picks
which styles to generate, and gets a personalized feed — cards grouped by style
and food pairing, each rateable 1–5 stars.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/wine/ranked?top_n=&user_id=&styles=` | Personalized ranking (popularity if no `user_id`) |
| `POST` | `/wine-events` | Log a wine rating (1–5 stars) |

### Dataset

- **[X-Wines](https://github.com/rogerioxavier/X-Wines)** (Full) — 100,646 wines, ~21M ratings. Raw CSVs live in `data/wine/` (git-ignored); `clean_wines.py` produces `clean_wines.csv` + `clean_ratings.csv`.

### Seed + try the demo

The wine catalog is seeded from the cleaned X-Wines CSV, then popularity stats are aggregated from the 21M ratings:

```bash
# One-shot: clean → reset wine tables → seed wines → compute avg_rating/n_ratings
./data/wine/clean_and_seed.sh

# (or run the steps individually)
python3 -m backend.db.reset_wines
python3 -m backend.db.wine.seed_wines
python3 -m backend.db.wine.compute_wine_stats   # populates the popularity prior
```

`compute_wine_stats` is what makes the ranking meaningful — without it every
wine ties on a flat popularity score.

Then, in the frontend, click **"Wine"** in the nav and hit **"Suggest me a wine"**.

You can also probe the API directly:

```bash
# Top 10 popular wines
curl "http://localhost:8000/wine/ranked?top_n=10" | jq

# Rate a wine
curl -X POST "http://localhost:8000/wine-events" \
     -H "Content-Type: application/json" \
     -d '{"user_id":1,"wine_id":100001,"event_type":"rate","rating":4.5}'
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

### Run the recipe pipeline

Run the recipe training steps directly (there is no longer a combined
`train_pipeline.sh` wrapper):

| Step | Script | Output |
|------|--------|--------|
| 1 | `python3 -m backend.db.seed_recipes` | 231k recipes in DB |
| 2 | `python3 -m backend.db.seed_ratings` | 1.1M ratings in DB |
| 3 | `python3 -m backend.ml.item_similarity` | `models/item_sim_matrix.npz` (sparse, top-50 per recipe) |
| 4 | `python3 -m backend.ml.train_cf` | `models/cf_model.pkl` (biased MF warm CF) |
| 5 | `python3 -m backend.ml.train_cb` | `models/cb_matrix.npz` (TF-IDF content-based) |
| 6 | `python3 -m backend.ml.evaluate` | `models/eval_results.json` (RMSE, Precision@K, NDCG@K) |

For wine, see **Wine recommender → Seed + try the demo** above for seeding. Then
build the model artifacts:

```bash
# 1. region rollup (2,160 appellations → ~107 parents)  [data processing]
python3 -m data.wine.region_rollup
# 2. content-based structured wine matrix
python3 -m backend.ml.wine.training.train_cb
# 3. (CF/ALS) — see backend/ml/wine/training/train_wine_als.py
#    inspect neighbors to sanity-check CB:  python3 -m data.wine.inspect_neighbors
```

After building, restart the backend — it picks up model files automatically.

---

## Docker

The recommended way to run everything:

```bash
./dev.sh                 # start (seeds the dev DB if needed, then brings the stack up)
./dev.sh --rebuild       # force image rebuild after dependency changes
# backend  → http://localhost:8000
# frontend → http://localhost:5173
```

Both services **hot-reload** in dev — edit a file and the change shows up
automatically, no rebuild:
- **Backend** runs under `uvicorn --reload` with the source bind-mounted.
- **Frontend** runs the **Vite dev server** with HMR (live source mount).

### How the dev setup is wired

`docker compose` auto-merges `docker-compose.override.yml` on top of
`docker-compose.yml`. The override swaps the production frontend (an nginx image
serving a static build) for the Vite dev server. That's why there are two
frontend Dockerfiles:
- `Dockerfile.frontend`     — production: build + serve with nginx.
- `Dockerfile.frontend.dev` — development: run `vite` with HMR.

### Running the production frontend build locally

To run the nginx static build instead of the dev server (i.e. ignore the
override), point Compose at the base file only:

```bash
docker compose -f docker-compose.yml up --build
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
    wine.py                  GET /wine/ranked (top-N popular); POST /wine-events
  services/
    scoring.py               Core ranking formula (RecipeScore dataclass)
    expiry.py                Urgency score (exponential decay)
    ingredient_match.py      Fuzzy ingredient overlap
    beta_updater.py          Daily preference learning job
    vision_agent.py          GPT-4o vision + ingredient canonicalization
    (wine ranking/pairing serving code removed — see Wine recommender above)
  ml/
    cold_start.py            Preference-seeded cold start CF (with fallback)
    item_similarity.py       Sparse item-item similarity matrix (training)
    train_cf.py              Biased MF training (scikit-surprise)
    train_cb.py              TF-IDF training (sklearn)
    serve_cf.py              CF serving — warm/cold auto-selection
    serve_cb.py              CB serving — cosine similarity at request time
    user_vector.py           Pantry → TF-IDF vector utility
    evaluate.py              RMSE, Precision@K, Recall@K, ablation
    wine/training/           Wine model training (offline; not imported by the app):
                               train_wine_als.py, train_cb.py, item_similarity.py,
                               build_wine_split.py, eval_*.py
  db/
    models.py                SQLAlchemy ORM (User, PantryItem, Recipe, UserEvent,
                               ShoppingListItem, Wine, WineEvent)
    database.py              Engine, session, init_db
    seed_dev.py              Dev seed: 20 recipes with tags/steps + demo pantry
    seed_recipes.py          Load Food.com CSV → Recipe table
    seed_ratings.py          Load Food.com ratings → UserEvent table
    reset_wines.py           Drop + recreate only the wine tables
    wine/
      seed_wines.py          Load clean_wines.csv → Wine table
      compute_wine_stats.py  Aggregate clean_ratings.csv → avg_rating/n_ratings
  canonicalizer/
    ingredient_map.py        Rule-based + fuzzy product name cleaner
    openfoodfacts.py         Barcode/name lookup via OFF API

frontend/src/
  App.tsx                    Router, auth guard, stale-user detection, nav
  api/client.ts              Axios client + all TypeScript types
  hooks/useUserId.ts         Persists user ID in localStorage
  index.css                  CSS custom properties + base styles
  api/wine.ts                Typed wrappers for the 5 wine endpoints
  components/
    ExpiryBadge.tsx          Color-coded days-remaining badge + urgency bar
    IngredientAutocomplete.tsx  Debounced autocomplete input backed by /pantry/suggest
    RecipeCard.tsx           Score ring + match ring, explainer, Cook→Rate flow, Buy missing button
    ScoreExplainer.tsx       4-component score breakdown bars (unavailable = grayed)
    VisionScanner.tsx        Photo scan → confirm → add to pantry
    WineCard.tsx             Path-B wine card with score breakdown, star rating
  pages/
    OnboardingPage.tsx       First-run: name, beta slider, diet tags
    PantryPage.tsx           Pantry management with expiry rows + scan button
    RecipeFeedPage.tsx       Ranked recipe feed with CF strategy banner + sort-by dropdown
    RecipeDetailPage.tsx     Full recipe + numbered steps
    BrowsePage.tsx           Search/filter all recipes (clickable → detail)
    WineForYouPage.tsx       "Suggest me a wine" feed (button → top 10 popular)
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
  explore_foodcom.ipynb      EDA notebook
  wine/
    download_wines.py        X-Wines download
    clean_wines.py           Raw X-Wines CSVs → clean_wines.csv + clean_ratings.csv
    clean_and_seed.sh        Clean → reset → seed → compute stats (one shot)

models/                      Trained artifacts (git-ignored)
  # recipes
  cf_model.pkl               Biased MF model (scikit-surprise SVD class)
  item_sim_matrix.npz        Sparse item-item similarity (top-50 per recipe)
  item_sim_recipe_ids.npy    Recipe ID index for similarity matrix
  cb_matrix.npz              TF-IDF recipe embeddings
  cb_vectorizer.pkl          Fitted TfidfVectorizer
  eval_results.json          Offline evaluation results
  # wine — currently present
  wine_als_model.npz         Confidence-weighted ALS factors + id maps
  wine_als_meta.json         Ranking metrics + hyperparams + timestamp
  wine_split/                Frozen leave-k-out train/test split
  # wine — produced by backend/ml/wine/training/ for the future CB features
  #         (not consumed by the app yet — serving is popularity-only for now)
  wine_cb_*                  TF-IDF wine embeddings (when train_cb.py is run)
  wine_sim_*                 Wine item-item cosine (when item_similarity.py is run)
```

---

## Team

TBD
---

## Dataset

[Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
— 231k recipes, 1.1M ratings, rich ingredient and tag features.
