# eXpairing — Complete Project Documentation

**Course:** Recommender Systems Workshop — Tel Aviv University  
**Instructor:** Dr. Rubi Boim  
**Stack:** Python · FastAPI · React · TypeScript · SQLite · scikit-surprise · sklearn

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prediction Target](#2-prediction-target)
3. [Architecture](#3-architecture)
4. [ML Design — CF-First System](#4-ml-design--cf-first-system)
5. [Data Features](#5-data-features)
6. [Candidate Generation](#6-candidate-generation)
7. [Cold Start Strategy](#7-cold-start-strategy)
8. [Beta Learning Loop](#8-beta-learning-loop)
9. [Diversity](#9-diversity)
10. [Evaluation](#10-evaluation)
11. [File Structure](#11-file-structure)
12. [Setup Guide — Quick Start](#12-setup-guide--quick-start)
13. [Setup Guide — Full ML Pipeline](#13-setup-guide--full-ml-pipeline)
14. [Setup Guide — Docker](#14-setup-guide--docker)
15. [API Reference](#15-api-reference)
16. [Frontend Pages](#16-frontend-pages)
17. [Testing](#17-testing)
18. [Configuration Reference](#18-configuration-reference)
19. [Demo Script](#19-demo-script)
20. [Recent Features](#20-recent-features)

---

## 1. Project Overview

eXpairing is a recipe recommender system that predicts user preference using collaborative filtering and adjusts rankings using domain-specific constraints (expiry urgency, ingredient availability).

### The problem

A user has a fridge with ingredients, some expiring soon. There are 230,000 recipes available. The system must predict which recipes the user will enjoy **and** surface ones that minimize food waste.

### What makes it a real recsys project

- **Items:** 230k+ recipes from Food.com
- **Users:** Food.com users with 1M+ ratings plus app users
- **Signals:** explicit ratings (1–5 stars) + implicit cook/skip events
- **Personalization:** per-user MF latent vectors, per-user β weight, diet tag filters
- **Cold start:** personalized item-based CF from first page load, with preference-score fallback ensuring CF is never zero even without a trained model
- **Online learning:** β drifts daily from revealed cooking behavior
- **Hybrid:** CF base + CB boost + domain adjustments
- **Recipe detail:** full ingredients and numbered step-by-step instructions from Food.com data
- **Ingredient autocomplete:** pantry form suggests canonical Food.com ingredient names as you type (prefix + substring matching, debounced)
- **Shopping list:** add missing ingredients from any recipe to a persistent buy-list; deduplication, check-off, and clear-purchased flows
- **Feed sort controls:** client-side sort by CF score, CB score, expiry urgency, or pantry match within the loaded 20 recipes

---

## 2. Prediction Target

The system models:

```
P(user will cook and enjoy recipe | user, recipe, pantry state)
```

Proxied by: estimated star rating [1,5], normalized to [0,1].

**Signals used:**

| Signal type | Example | Used by |
|------------|---------|---------|
| Explicit | Star rating 1–5 | matrix factorization training (primary) |
| Implicit — cook | Cooked with n_missing=0 | β updater (revealed preference) + MF augmentation |
| Implicit — skip | Dismissed from feed | 7-day feed exclusion |
| Contextual | Pantry expiry dates | Expiry urgency adjustment |

Both explicit and implicit signals are used — complementary, not competing.
Explicit ratings give calibrated preference strength. Implicit cook events
reveal true behavior more frequently and reflect real decision-making.
Cook events are also converted to synthetic ratings (`max(3.0, 4.0 − min(n_missing,3) × 0.3)`)
and merged into matrix factorization training data when no explicit rating exists, accelerating warm-up
for users who cook frequently but rarely rate.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser  (React + TypeScript)                                  │
│  Onboarding · Pantry · Feed · Detail · Browse · Profile · List  │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP / REST
┌──────────────────────▼──────────────────────────────────────────┐
│  FastAPI  (Python)                                              │
│  /pantry · /recipes/ranked · /vision · /users · /shopping       │
└────┬───────────┬──────────┬──────────────┬──────────────────────┘
     │           │          │              │
     ▼           ▼          ▼              ▼
 SQLite DB    Scoring    Vision Agent   Beta Updater
 (ORM)        Service    (GPT-4o)       (daily batch)
              │
    ┌─────────┴──────────────────────────────────────────┐
    │  Two-stage pipeline                                │
    │                                                    │
    │  Stage 1: Candidate generation                     │
    │    diet_tags filter + popularity cap               │
    │    230k recipes → ~2000 candidates                 │
    │                                                    │
    │  Stage 2: Ranking (CF-first)                       │
    │    serve_cf.py  →  cold_start.py  or  biased MF          │
    │    serve_cb.py  →  cosine similarity               │
    │    scoring.py   →  blend + sort                    │
    └─────────┬──────────────────────────────────────────┘
              │
    ┌─────────┴──────────────────────────────────────────┐
    │  Trained model artifacts  (models/)                │
    │  cf_model.pkl          biased MF model                   │
    │  item_sim_matrix.npz   sparse item-item CF matrix  │
    │  cb_matrix.npz         TF-IDF recipe embeddings    │
    │  cb_vectorizer.pkl     TF-IDF vectorizer           │
    └────────────────────────────────────────────────────┘
```

### Request flow for GET /recipes/ranked

```
1. Load user profile (beta, diet_tags)
2. Load pantry items (ingredient + expiry_date)

Stage 1 — Candidate generation:
3. Filter by diet_tags (hard constraint)
4. Cap at 2000 by avg_rating descending
   → 230,000 recipes → ~2,000 candidates

Stage 2 — Ranking:
5. Count user ratings → select CF strategy:
      n_ratings < 5  →  item-based cold start CF (preference-seeded)
      n_ratings ≥ 5  →  biased matrix factorization (personalized vectors)
   CF scores are always non-zero — preference-score fallback when no model files
6. Compute CB scores: cos(pantry_vector, recipe_vectors)
7. Compute expiry urgency: exp(-k·days_to_expiry) per ingredient
8. Compute ingredient match ratio (fuzzy overlap)
9. Blend: final = γ·CF + δ·CB + α·expiry + β·match
10. Sort descending → return top-N with score breakdown
```

---

## 4. ML Design — CF-First System

### The architecture: CF predicts, domain adjusts

The system is structured as **CF with domain-specific adjustments** — not as four equal features:

```
base_score  = CF(user, recipe)              ← predicts user preference

final_score = base_score
            + CB_boost                      ← ingredient profile affinity
            + domain_adjustments            ← expiry urgency + availability
```

Formally:

```
final_score = γ · cf_score          (CF base — highest single weight)
            + δ · cb_score          (CB ingredient profile boost)
            + α · expiry_urgency    (domain: waste minimization)
            + β · match_ratio       (domain: ingredient availability)
```

**Default weights:**

| Weight | Default | Role |
|--------|---------|------|
| γ | 0.35 | CF base — highest single weight, reflecting CF primacy |
| α | 0.35 | Expiry urgency domain adjustment |
| β | 0.20 | Ingredient match (per-user, learned from behavior) |
| δ | 0.10 | CB ingredient profile boost |

β is per-user and learned from revealed cooking behavior via beta_updater.py. When CB is unavailable (model not trained), δ=0.10 redistributes to γ and α.

### Component 1 — Collaborative Filtering (the core model)

**Algorithm:** Biased matrix factorization (scikit-surprise).

Decomposes the sparse user-item matrix R into latent factors:

```
R ≈ P · Q^T + bias

predicted_rating(u, r) = dot(P[u], Q[r]) + bias_u + bias_r + global_mean
```

- P ∈ R^(n_users × n_factors): user latent vectors
- Q ∈ R^(n_recipes × n_factors): recipe latent vectors
- Default: n_factors=50, n_epochs=20

**Data sparsity:** The Food.com matrix is ~99.998% empty. biased MF handles sparsity via low-rank approximation — the model generalizes from ~1M observed ratings to predict all missing ones.

**Trained on explicit signals:** star ratings 1–5 only. Implicit signals (cook, skip) feed separate components.

### Component 2 — Item-Based CF (cold start + similarity)

**Algorithm:** Item-item cosine similarity on mean-centered rating matrix.

```
sim(i, j) = cos(R_T[i], R_T[j]) = (R_T[i] · R_T[j]) / (‖R_T[i]‖ · ‖R_T[j]‖)
```

Where R_T = recipe×user matrix, mean-centered per user (removes rating-scale bias).

Two recipes are similar if the **same users** rated them highly. No content information enters this computation.

**Why item-based over user-based:**
- More robust to sparsity (item vectors are denser than new-user vectors)
- Item similarity is more stable over time (recipes don't change taste)
- Enables cold start: new users scored via item neighborhoods

**Sparse implementation:** The full 231k×196k pivot is infeasible (44 billion cells). The item similarity matrix is built entirely with scipy sparse operations — filters to recipes with ≥5 ratings (~51k), builds sparse CSR, mean-centers in-place, L2-normalizes, and computes cosine in CHUNK_SIZE-row batches without ever materializing the full n×n output. Result: ~51k×51k sparse matrix, top-50 neighbors per recipe, 2.5M non-zero entries.

**Fallback when no model files:** cold_start.py computes raw preference scores from tag/pantry overlap (scores in [0,1], never normalized to avoid clustering near 1.0). CF is never zero even on a fresh install.

### Component 3 — Content-Based Filtering (the boost)

**Item profiles:** TF-IDF vectors over ingredient tokens (unigrams + bigrams, vocab size 20,000).

**User profile:** Current pantry ingredients joined as a single document, vectorized with the same fitted TfidfVectorizer.

**Similarity:**
```
cb_score(u, r) = cos(pantry_vector(u), recipe_vector(r))
```

Captures cuisine affinity: a pantry with miso, soy sauce, and sesame oil naturally cosine-matches Japanese recipes — even without explicit preference data.

**Difference from CF:** CB uses item content (ingredients). CF uses user interaction data (ratings, co-cooking). Together they form the hybrid.

### Component 4 — Expiry Urgency (domain adjustment)

```
urgency(days) = exp(-k · max(days, 0))    k = ln(2) / half_life
```

Default half-life = 3 days. This is a **domain constraint, not a preference prediction.** It adjusts rankings to surface waste-reducing recipes without overriding the CF preference model.

### Component 5 — Ingredient Match (domain adjustment)

Fuzzy ingredient overlap (rapidfuzz partial_ratio, threshold 75) between recipe ingredients and pantry contents. Returns match_ratio and the missing ingredient list for UI display. β scales this constraint per user.

---

## 5. Data Features

### User features
| Feature | Source | Used by |
|---------|--------|---------|
| Rating history | POST /events (rate) | matrix factorization training (explicit) |
| Cook history + n_missing | POST /events (cook) | β updater (implicit) |
| Skip history | POST /events (skip) | Future negative signal |
| `beta` | Onboarding slider + daily updater | match_ratio weight |
| `diet_tags` | Onboarding | Candidate filter + cold-start seeds |
| `is_warm` | Auto-computed at ≥5 ratings | CF strategy selection |

### Item (recipe) features
| Feature | Source | Used by |
|---------|--------|---------|
| Rating vectors | 1M Food.com ratings | MF training, item-item sim |
| `ingredients_csv` | Food.com | TF-IDF (CB), ingredient match |
| `tags_csv` | Food.com | Diet filter, cold-start seeds |
| `avg_rating` | Aggregated from ratings | Candidate ranking, display |
| `minutes` | Food.com | Display |
| `steps_json` | Food.com | Recipe detail page |
| `description` | Food.com | Recipe detail page |
| `n_steps` | Food.com | Recipe detail page |

### Pantry features
| Feature | Source | Used by |
|---------|--------|---------|
| `ingredient` | Manual / vision scan | Ingredient match, CB vector |
| `expiry_date` | Manual / OCR / vision | Expiry urgency score |

---

## 6. Candidate Generation

Large-scale recsys follows a two-stage pipeline. Scoring all 230k recipes per request in real-time is infeasible. Candidate generation reduces the search space first.

### Stage 1: Candidate generation (230k → ~2000)

```
Step 1: Filter by diet_tags (hard constraint)
        User marked "vegetarian" → remove all non-vegetarian recipes
        Graceful fallback: relaxes tags one-by-one if corpus too small

Step 2: Order by avg_rating descending, cap at 2000
        Keeps the most-rated, highest-quality recipes as candidates
```

This is a deliberate design decision. The top-2000 by rating captures the recipes most likely to appear in any user's ranked list. Domain adjustments (expiry, pantry match) then re-rank within this candidate set.

### Stage 2: Scoring (2000 → top-N)

All 2000 candidates are scored by the CF-first formula and top-N returned.

**Future improvement:** Pre-filter by CB similarity to pantry before scoring, further reducing candidates to a highly relevant subset.

---

## 7. Cold Start Strategy

### The problem

New users have no rating history. Biased MF cannot produce a user vector. The naive fallback — global popularity — is identical for everyone and ignores all onboarding information.

### Why our cold start is still Collaborative Filtering

A common misunderstanding: using diet_tags and pantry to infer preferences looks like content-based filtering. It is not. The distinction:

**Content-based:** similarity computed from item attributes (ingredients, tags). No user data.

**Our cold start:** similarity matrix computed entirely from **user rating co-occurrence patterns**. `sim(i,j) = cos(R_T[i], R_T[j])` where R_T is the mean-centered rating matrix. Diet_tags and pantry select **anchor recipes** (pseudo-interactions) — they do not enter the similarity computation itself.

Therefore: **this is CF with inferred preferences**, not content-based filtering. The community's co-rating patterns drive recommendations; we substitute preference-inferred anchors for the missing rating history.

### Algorithm

**Step 1 — Seed selection (preference inference)**
```
seed_score(r) = tag_weight · tag_match(r, user_tags)
              + pantry_weight · pantry_overlap(r, pantry)
```
A vegetarian user with eggs and milk → vegetarian egg-and-milk recipes as seeds.
A vegan user with tofu → entirely different seeds → different CF scores downstream.

**Step 2 — Seed diversification**
Cap each primary cuisine/tag at max_per_tag=5 seeds. Prevents all 30 seeds being Italian (echo chamber). Inspired by MMR (Maximal Marginal Relevance).

**Step 3 — Item-CF scoring (when model available)**
```
score(candidate) = mean cosine_similarity(candidate, seed_i)
                   for seed_i in seed_set
```
Same item-based CF formula — anchored on pseudo-interactions.

**Step 3 (fallback — no model files)**
When no `item_sim_matrix.npz` exists, scores are computed directly from tag + pantry overlap:
```
score(candidate) = (1 - pantry_weight) · tag_match(candidate, user_tags)
                 + pantry_weight · pantry_overlap(candidate, pantry)
```
Scores are raw [0,1] values — **never normalized** (normalization clusters all scores near 1.0, flattening the ranking). This guarantees CF is never zero from first boot.

**Step 4 — Automatic transition to biased MF**
At 5 ratings with a trained biased MF model, serve_cf.py switches to biased MF. No manual flag.

### Coverage table

| Scenario | Strategy | Personalized? |
|----------|---------|---------------|
| 0 ratings, no tags, no model | Tag+pantry preference scores | Weakly |
| 0 ratings, tags set | Preference-seeded item-CF (or fallback) | Yes — by taste |
| 0 ratings, tags + pantry | Preference + pantry seeded | Yes — taste + pantry |
| ≥ 5 ratings, MF model | biased matrix factorization | Yes — from history |

---

## 8. Beta Learning Loop

β is the per-user weight controlling how much missing ingredients penalize a recipe. It captures the **waste-aversion preference**.

### Stated vs revealed preference

Users set β on the onboarding slider ("Discover new recipes" → "Use what I have"). Stated preferences are aspirational — people claim to be zero-waste but cook with whatever looks good. We detect this gap from implicit signals:

```
revealed_beta = 1.0 - (avg_n_missing_per_cook / MAX_MISSING_NORMALIZER)
```

### Drift formula (exponential moving average)

```
new_beta = (1 - LEARNING_RATE) · current_beta
         + LEARNING_RATE       · revealed_beta
```

LEARNING_RATE = 0.15. β moves 15% toward revealed preference per daily run. Never jumps — one unusual session doesn't dominate.

### Convergence

After 30 consistent days, β converges within 5% of revealed_beta.
Proven by: `test_converges_to_zero_waste` and `test_converges_to_permissive`.

```bash
# Preview changes without writing
python3 -m backend.services.beta_updater --dry-run

# Apply updates
python3 -m backend.services.beta_updater

# Production cron (3am daily)
# 0 3 * * * cd /app && python3 -m backend.services.beta_updater
```

---

## 9. Diversity

To avoid repetitive recommendations (e.g., all pasta):

**Score calibration (scoring.py → `_calibrate`):**
Each of the four score components (CF, CB, expiry, match) is min-max normalized to [0,1] across the full candidate pool before the weighted blend. Without calibration, a wide-ranging expiry signal (spanning 0.0–0.9) would dominate a narrow CF signal (spanning 0.4–0.6) regardless of weights.

**MMR reranking (scoring.py → `_mmr_rerank`):**
After scoring and sorting, the top 3×top_n candidates pass through Maximal Marginal Relevance selection:
```
MMR(r) = λ · final_score(r) − (1−λ) · max_sim(r, already_selected)
```
Similarity = ingredient Jaccard. λ=0.7 keeps relevance primary. Selection is greedy: always pick the candidate maximizing the MMR score. The first item selected is always the highest-scored recipe.

**Implemented diversity mechanisms:**
- Score calibration across all candidates
- MMR ingredient-diversity reranking (λ=0.7)
- Cold-start seed diversification caps 5 seeds per cuisine/tag
- Diet tag filter removes entire ineligible categories
- 7-day skip exclusion removes recently dismissed recipes

**Skip exclusion (recipes.py):**
Recipes dismissed via "Skip" are excluded from the candidate pool for 7 days. The cutoff is a datetime comparison on `UserEvent.created_at`.

---

## 10. Evaluation

### Running evaluation

```bash
# Quick evaluation (synthetic data, no Food.com needed)
python3 -m backend.ml.evaluate

# Full evaluation (after seed_ratings.py)
python3 -m backend.ml.evaluate --full

# Ablation study only
python3 -m backend.ml.evaluate --ablation

# Lifecycle simulation — NDCG@10 vs n_ratings
python3 -m backend.ml.evaluate --lifecycle

# Weight grid search — find optimal (γ, α)
python3 -m backend.ml.evaluate --tune
```

Results saved to `models/eval_results.json`.

### Metric 1 — Rating prediction accuracy (RMSE / MAE)

Evaluates how well the CF model predicts held-out ratings.
Test set = last 20% of each user's ratings.

| Model | RMSE | Notes |
|-------|------|-------|
| Global mean baseline | ~1.12 | Predict dataset mean for everyone |
| Per-user mean baseline | ~1.05 | Predict each user's mean |
| **Biased MF (n_factors=50)** | **~0.82** | **Our CF model** |
| **Improvement** | **~27%** | **Over global mean baseline** |

*Exact values depend on dataset. Run `python3 -m backend.ml.evaluate` for current numbers.*

### Metric 2 — Ranking quality (Precision@K / Recall@K)

Evaluates whether the top-K ranked recipes are ones the user actually likes.
Relevant = rating ≥ 4.0 stars.

*Run `python3 -m backend.ml.evaluate --full` for current numbers.*

### Metric 3 — NDCG@K (graded ranking quality)

NDCG (Normalized Discounted Cumulative Gain) rewards putting the highest-rated items at rank 1:

```
DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i + 1)
NDCG@K = DCG@K / IDCG@K    (normalized by ideal ordering)
rel_i  = (rating - 1) / 4   (scaled to [0,1])
```

Unlike Precision@K (binary relevant/not), NDCG reflects how well we order items by preference strength. A system that puts a 5★ recipe at rank 1 scores higher than one that puts it at rank 5, even if both are "relevant".

### Metric 4 — Lifecycle simulation

Simulates a user acquiring ratings from 0 → N and measures NDCG@10 at each step. Validates that the soft CF blend (α = min(n/5, 1.0)) ramps smoothly from cold-start to warm biased MF without a sudden quality jump at the threshold.

### Metric 5 — Weight grid search

Grid-searches (γ, α) combinations and reports NDCG@10 per configuration. Confirms or suggests updates to DEFAULT_GAMMA and DEFAULT_ALPHA in scoring.py. If the best config beats the current defaults by > 0.005 NDCG, update the constants.

### Metric 3 — Ablation study (Precision@10)

Quantifies each component's contribution:

| Model | Description | Precision@10 |
|-------|-------------|-------------|
| CF only | MF predictions only | baseline |
| CB only | TF-IDF cosine sim only | lower |
| Domain only | Expiry + match only | lower |
| **Full hybrid** | **All components** | **highest** |

**Conclusion:** CF provides the strongest predictive signal. The hybrid improves over any single component, especially for new or obscure recipes where CF has sparse data.

---

## 11. File Structure

```
smartrecipes/
│
├── backend/
│   ├── main.py                     FastAPI app, CORS, router registration
│   ├── routers/
│   │   ├── pantry.py               GET/POST/PUT/DELETE /pantry/{user_id}
│   │   │                             GET /pantry/suggest (ingredient autocomplete)
│   │   ├── recipes.py              GET /recipes/ranked, GET /recipes/search,
│   │   │                             GET /recipes/{id}, POST /events
│   │   ├── users.py                GET/POST/PUT /users, GET /users/{id}/stats
│   │   ├── vision.py               GET /vision/mock, POST /vision/scan,
│   │   │                             POST /vision/confirm/{user_id}
│   │   └── shopping.py             GET/POST/PATCH/DELETE /shopping/{user_id}
│   │                                 deduplication, check-off, clear-purchased
│   ├── services/
│   │   ├── scoring.py              CF-first ranking formula, RecipeScore dataclass
│   │   ├── expiry.py               Urgency score (exponential decay)
│   │   ├── ingredient_match.py     Fuzzy ingredient overlap + missing list
│   │   ├── beta_updater.py         Daily β drift: stated → revealed preference
│   │   └── vision_agent.py         GPT-4o vision + product canonicalization
│   ├── ml/
│   │   ├── train_cf.py             Biased MF training + 3-fold CV RMSE
│   │   ├── train_cb.py             TF-IDF ingredient embedding training
│   │   ├── item_similarity.py      Sparse item-item cosine similarity matrix
│   │   │                             (fully sparse; chunks 231k×196k problem)
│   │   ├── cold_start.py           Preference-seeded cold-start CF;
│   │   │                             fallback preference scores when no model
│   │   ├── serve_cf.py             Auto warm/cold CF selection
│   │   ├── serve_cb.py             CB cosine similarity at request time
│   │   ├── user_vector.py          Pantry → TF-IDF vector utility
│   │   └── evaluate.py             RMSE, Precision@K, Recall@K, ablation
│   ├── db/
│   │   ├── models.py               ORM: User, PantryItem, Recipe, UserEvent, ShoppingListItem
│   │   │                             Recipe has steps_json, description, n_steps
│   │   ├── database.py             Engine, SessionLocal, init_db
│   │   ├── seed_dev.py             20 recipes with varied tags + steps + demo pantry
│   │   ├── seed_recipes.py         Load Food.com CSV → Recipe table (incl. steps)
│   │   └── seed_ratings.py         Load Food.com ratings → UserEvent table
│   └── canonicalizer/
│       ├── ingredient_map.py       Rule-based + fuzzy product name cleaner
│       └── openfoodfacts.py        Barcode/name lookup via OFF API
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── playwright.config.ts        E2E test configuration
│   ├── playwright-demo.config.ts   Demo recording configuration (video: on)
│   ├── e2e/
│   │   ├── expairing.spec.ts     63 Playwright E2E tests
│   │   └── demo.spec.ts            Full feature demo recording script
│   └── src/
│       ├── App.tsx                 Router, stale-user detection, nav
│       ├── index.css               CSS custom properties + base styles
│       ├── api/client.ts           Axios client + TypeScript types
│       ├── hooks/useUserId.ts      Persist user ID in localStorage
│       ├── components/
│       │   ├── ExpiryBadge.tsx     Color-coded badge + urgency bar
│       │   ├── IngredientAutocomplete.tsx  Debounced autocomplete backed by /pantry/suggest
│       │   ├── RecipeCard.tsx      Score ring + match ring, explainer, Cook→Rate, Buy missing
│       │   ├── ScoreExplainer.tsx  4-bar breakdown (unavailable bars grayed)
│       │   └── VisionScanner.tsx   Photo scan → confirm → pantry
│       └── pages/
│           ├── OnboardingPage.tsx  First run: name, β slider, diet tags
│           ├── PantryPage.tsx      Pantry management + autocomplete + scan button
│           ├── RecipeFeedPage.tsx  Ranked feed + CF strategy banner + sort-by dropdown
│           ├── RecipeDetailPage.tsx Full recipe: ingredients + step instructions
│           ├── BrowsePage.tsx      Search/filter corpus (names link to detail)
│           ├── ProfilePage.tsx     β + diet tags + CF progress bar
│           └── ShoppingListPage.tsx  Buy-list: check off items, clear purchased
│
├── tests/
│   ├── conftest.py                 sys.path setup
│   ├── test_scoring.py             51 unit tests
│   ├── test_beta_updater.py        32 unit tests
│   ├── test_cf.py                  29 unit tests
│   ├── test_cold_start.py          33 unit tests (incl. fallback score assertions)
│   ├── test_vision_agent.py        40 unit tests
│   ├── test_evaluate.py            27 unit tests
│   ├── test_shopping.py            20 unit tests (shopping list CRUD + deduplication)
│   └── test_ml_behavior.py         38 behavioral integration tests
│                                     — requires backend at localhost:8000
│
├── data/
│   ├── download_foodcom.py         Kaggle download script
│   ├── train_pipeline.sh           Full 6-step pipeline
│   └── explore_foodcom.ipynb       EDA notebook
│
├── models/                         Trained artifacts (git-ignored)
│   ├── cf_model.pkl
│   ├── cf_meta.json
│   ├── item_sim_matrix.npz         Sparse ~51k×51k, top-50 per recipe
│   ├── item_sim_recipe_ids.npy
│   ├── item_sim_meta.json
│   ├── cb_matrix.npz
│   ├── cb_recipe_ids.npy
│   ├── cb_vectorizer.pkl
│   ├── cb_meta.json
│   └── eval_results.json
│
├── docker-compose.yml
├── Dockerfile.backend
├── frontend/Dockerfile.frontend
├── requirements.txt
├── README.md
└── EXPAIRING.md
```

---

## 12. Setup Guide — Quick Start

### Prerequisites

- Python 3.9+
- Node 18+

### 1. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Seed dev database

```bash
python3 -m backend.db.seed_dev
```

Creates `fridge2fork.db` with 20 recipes (varied dietary tags, realistic steps, diverse avg_ratings) and a demo user with 10 pantry items:

```
Seeded 20 recipes.
Created user id=1 with 10 pantry items.
  milk       expires in 2 days
  tomatoes   expires in 2 days
  ...
```

### 3. Train CB model (optional, ~2 seconds on dev data)

```bash
python3 -m backend.ml.train_cb
```

### 4. Start backend

```bash
python3 -m uvicorn backend.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
# Health:     http://localhost:8000/health
```

### 5. Start frontend

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

### 6. Run tests

```bash
# Unit tests (no backend needed)
python3 -m pytest tests/ -v --ignore=tests/test_ml_behavior.py

# Behavioral integration tests (requires backend running)
python3 -m pytest tests/test_ml_behavior.py -v

# E2E Playwright tests (requires backend + frontend running)
cd frontend && npx playwright test --reporter=list
```

---

## 13. Setup Guide — Full ML Pipeline

### Step 1: Kaggle API key

1. https://www.kaggle.com → Account → Create New API Token
2. Save `kaggle.json` to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

### Step 2: Download Food.com dataset

```bash
python3 -m data.download_foodcom
```

Downloads:
- `data/RAW_recipes.csv` — ~230MB, 231k recipes (with steps and descriptions)
- `data/RAW_interactions.csv` — ~55MB, 1.1M user ratings

### Step 3: Full training pipeline

```bash
chmod +x data/train_pipeline.sh
./data/train_pipeline.sh          # full dataset
./data/train_pipeline.sh 10000   # 10k recipes for faster dev
```

| Step | Script | Description | Notes |
|------|--------|-------------|-------|
| 1 | `seed_recipes.py` | 231k recipes → DB (with steps) | ~3 min |
| 2 | `seed_ratings.py` | 1M ratings → DB | ~5 min |
| 3 | `item_similarity.py` | Sparse item-item CF matrix | ~4 min; filters to ~51k recipes with ≥5 ratings |
| 4 | `train_cf.py` | Biased MF + 3-fold CV | ~5 min |
| 5 | `train_cb.py` | TF-IDF embeddings | ~2 min |
| 6 | `evaluate.py` | RMSE + Precision@K | ~3 min |

### Step 4: Vision scan (optional)

```bash
export OPENAI_API_KEY=sk-...
```

---

## 14. Setup Guide — Docker

```bash
docker-compose up --build
```

- Frontend → http://localhost:5173
- Backend  → http://localhost:8000
- Swagger  → http://localhost:8000/docs

With OpenAI key:
```bash
OPENAI_API_KEY=sk-... docker-compose up --build
```

---

## 15. API Reference

### Users

| Method | Endpoint | Description |
|--------|---------|-------------|
| POST | `/users` | Create user (onboarding) — returns 201 |
| GET | `/users/{id}` | Get profile |
| PUT | `/users/{id}` | Update β and diet tags |
| GET | `/users/{id}/stats` | n_ratings, warm_cf_progress_pct, is_warm, n_cooked |

### Pantry

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/pantry/suggest?q=eg&limit=8` | Autocomplete: prefix+substring match against recipe corpus vocabulary |
| GET | `/pantry/{user_id}` | List all items, sorted by expiry |
| POST | `/pantry/{user_id}` | Add one item — returns 201 |
| POST | `/pantry/{user_id}/bulk` | Add multiple items |
| PUT | `/pantry/{user_id}/{item_id}` | Update expiry/quantity |
| DELETE | `/pantry/{user_id}/{item_id}` | Remove one item |
| DELETE | `/pantry/{user_id}` | Clear entire pantry |

`/pantry/suggest` returns prefix matches first, then substring matches, up to `limit`. Vocabulary is lazily built from all `ingredients_csv` values in the Recipe table on first call.

### Shopping List

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/shopping/{user_id}` | List all items (id, ingredient, source recipe, is_checked) |
| POST | `/shopping/{user_id}` | Add ingredients — deduplicates; returns `{added, skipped}` |
| PATCH | `/shopping/{user_id}/{item_id}` | Toggle `is_checked` |
| DELETE | `/shopping/{user_id}/{item_id}` | Remove one item |
| DELETE | `/shopping/{user_id}` | Clear entire list (or only checked items via `?checked_only=true`) |

**POST /shopping/{user_id} request body:**
```json
{
  "ingredients": ["bread", "cream cheese"],
  "recipe_id": 42,
  "recipe_name": "Bagels"
}
```

**Response:**
```json
{
  "added": [{"id": 1, "ingredient": "bread", "source_recipe_id": 42, "source_recipe_name": "Bagels", "is_checked": false}],
  "skipped": ["cream cheese"]
}
```

### Recipes

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/recipes/ranked?user_id=1&top_n=20` | CF-first ranked feed |
| GET | `/recipes/search?q=pasta&tag=vegetarian&limit=40` | Browse corpus |
| GET | `/recipes/{id}` | Single recipe detail (ingredients, steps, description) |
| POST | `/events` | Log cook / skip / rate — returns 201 |

**Response — /recipes/ranked (one item):**
```json
{
  "recipe_id": 42,
  "recipe_name": "French toast",
  "final_score": 0.7341,
  "cf_score": 0.61,
  "cb_score": 0.44,
  "expiry_urgency": 0.89,
  "match_ratio": 0.75,
  "matched_ingredients": ["eggs", "milk", "butter"],
  "missing_ingredients": ["bread"],
  "total_ingredients": 4,
  "tags": ["breakfast", "quick", "vegetarian"],
  "minutes": 15,
  "avg_rating": 4.3,
  "cf_strategy": "item_based_cold_start"
}
```

`cf_strategy`:
- `"biased_mf"` — ≥5 ratings + biased MF model trained, full personalization active
- `"item_based_cold_start"` — new user or no MF model; personalized via preference seeds
- `"none"` — no CF model and no preference signal

**Response — /recipes/{id}:**
```json
{
  "id": 42,
  "name": "French toast",
  "ingredients": ["eggs", "milk", "butter", "bread"],
  "tags": ["breakfast", "quick"],
  "minutes": 15,
  "avg_rating": 4.3,
  "description": "Classic French toast...",
  "steps": [
    "Whisk eggs with milk.",
    "Dip bread slices.",
    "Cook on buttered pan until golden."
  ],
  "n_steps": 3
}
```

**POST /events:**
```json
{
  "user_id": 1,
  "recipe_id": 42,
  "event_type": "rate",
  "rating": 4,
  "n_missing": 1
}
```

`event_type`:
- `"cook"` — implicit signal, n_missing feeds β updater
- `"skip"` — implicit signal, future negative reranking
- `"rate"` — explicit signal, feeds matrix factorization training; 5 ratings → is_warm=True

### Vision

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/vision/mock` | Demo scan (no API key needed) |
| POST | `/vision/scan` | Real GPT-4o fridge photo scan |
| POST | `/vision/confirm/{user_id}` | Bulk-add confirmed items to pantry |

**POST /vision/confirm request body:**
```json
{
  "items": [
    {"ingredient": "milk", "expiry_date": "2026-06-01", "raw_name": "Tnuva 3% Milk", "quantity": "500ml"},
    {"ingredient": "eggs", "expiry_date": "2026-06-05", "raw_name": "Free Range Eggs", "quantity": "6"}
  ]
}
```

Items with `expiry_date: null` are rejected (422). The `expiry_date` is parsed to a Python `date` before DB insert.

---

## 16. Frontend Pages

### Onboarding (`/`)

Creates user. Collects: name, β slider ("Discover new recipes" ↔ "Use what I have"), diet tags. β self-corrects from behavior — onboarding is a starting point.

### Pantry (`/pantry`)

Items sorted by expiry. Row backgrounds: white → amber → red by urgency. Each row shows the ExpiryBadge (e.g. "2d left") and a small urgency progress bar. Summary pills show expiring-soon count. Demo scan simulates the full vision pipeline.

The ingredient input uses `IngredientAutocomplete`: typing 2+ characters fires a debounced request to `GET /pantry/suggest`. The dropdown lists up to 8 suggestions (prefix matches first), navigable by keyboard (↑/↓/Tab/Enter) or mouse. Selecting a suggestion fills the field and dismisses the dropdown without blocking form submission.

### Recipe Feed (`/feed`)

CF strategy banner:
- Blue = cold start active ("Personalized for you — rate 5 recipes to unlock full personalization")
- Green = biased MF active ("Personalized from your history")

Subtitle: *X recipes ranked by collaborative filtering (CF) · expiry urgency · pantry match* — CF is listed first to reflect its highest weight in the scoring formula.

**Sort-by dropdown:** Allows re-sorting the 20 loaded recipes client-side by:
- Total score (default — preserves server ranking)
- CF score
- CB score
- Expiry urgency
- Pantry match

Each RecipeCard shows two circular rings:
- **Score ring** (blue) — overall final_score
- **Match ring** (green/amber/gray) — pantry match_ratio

Expandable **"Why this recipe?"** shows all 4 score component bars:
- **Expiry urgency** — grayed if no expiring items
- **Ingredient match** — grayed if empty pantry
- **Community score (CF)** — shows CF strategy badge
- **Profile match (CB)** — grayed if CB model not trained

Cook → star rating → feed refreshes. Skip removes card. Recipe names link to detail page.

**"＋ Buy missing" button** — adds the recipe's missing ingredients to the shopping list in one click. Button label changes to "Added to list" after the first click.

### Recipe Detail (`/recipe/:id`)

Full recipe view: name, badges (time, steps, avg rating, dietary tags), description paragraph, bulleted ingredient list, numbered step-by-step instructions. Back button returns to originating page. Populated from Food.com `steps` and `description` fields.

### Browse (`/browse`)

Debounced text search + tag filter pills. Search matches by name or ingredient. Results sorted by avg_rating. Ingredient lists expandable per card. Recipe names link to detail page.

### Profile (`/profile`)

- β slider ("Discover new recipes" ↔ "Use what I have")
- Personalization status card: Cold start badge + progress bar (x/5 ratings) or Personalized badge
- Cook count + rating count
- Diet tag checkboxes (save triggers re-ranking on next feed load)

### Shopping List (`/list`)

Persistent buy-list across sessions. Each row shows the ingredient name and the source recipe it was added from. Checkboxes mark items as purchased (struck-through). "Clear purchased" removes all checked items. Items are deduplicated — adding "eggs" from a second recipe keeps only the first entry.

---

## 17. Testing

### Unit tests (no backend required)

```bash
python3 -m pytest tests/ --ignore=tests/test_ml_behavior.py -v
# ~232 tests, ~10 seconds
```

| File | Tests | Covers |
|------|-------|--------|
| `test_scoring.py` | 51 | Urgency decay, fuzzy match, ranking, weight redistribution |
| `test_beta_updater.py` | 32 | β math, DB integration, convergence after 30 cycles |
| `test_cf.py` | 29 | biased MF routing, item-sim, warm/cold threshold, transition |
| `test_cold_start.py` | 33 | Seed selection, diversification, **fallback preference scores** |
| `test_vision_agent.py` | 40 | Brand stripping, fuzzy match, mock scan, OFF extractor |
| `test_evaluate.py` | 27 | RMSE math, Precision@K logic, train/test split |
| `test_shopping.py` | 20 | Shopping list CRUD, deduplication (`added`/`skipped`), check-off, clear |

### Behavioral integration tests (requires backend at localhost:8000)

```bash
python3 -m pytest tests/test_ml_behavior.py -v
# 37 tests pass + 1 skipped (real vision scan — needs OPENAI_API_KEY + test image)
```

| Group | Tests | What they verify |
|-------|-------|-----------------|
| `TestInitialState` | 9 | Feed sorted, all scores in [0,1], zero urgency/match with empty pantry, CF differentiated |
| `TestPantryEffect` | 7 | Adding items raises match_ratio; near-expiry > far urgency; deterministic; removing lowers coverage |
| `TestCFProgression` | 8 | is_warm flips at exactly 5 ratings; progress counter; biased MF activates with trained model |
| `TestCBAndBeta` | 4 | CB in range; diet tags don't starve feed; high-beta top-5 has higher pantry match |
| `TestScoreMath` | 3 | final_score within component bounds; urgency surfaces relevant recipes; matched ⊆ pantry |
| `TestVision` | 7 | Mock structure; confirm→pantry (date parsing fixed); confirm→feed match; null-expiry rejection |

Conditional tests:
- `@skipif(not CF_MODEL_EXISTS)` — biased MF strategy tests only run when `models/cf_model.pkl` exists
- `@skipif(not OPENAI_API_KEY)` — real vision test requires key and test image at `/tmp/test_food.jpg`

### End-to-end Playwright tests (requires backend + frontend running)

```bash
cd frontend
npx playwright test --reporter=list
# 63 tests, ~45 seconds
```

| Group | Tests | Covers |
|-------|-------|--------|
| Onboarding | 5 | First-run flow, slider, diet tags, disabled state |
| Navigation | 3 | All links present, correct routes |
| Pantry | 9 | Add/delete, expiry badge, autocomplete (keyboard + mouse), backend suggest API |
| Recipe Feed | 10 | Score rings, CF banner, cook→rate, skip, sort-by dropdown, subtitle |
| Recipe Detail | 4 | Full recipe view, steps, navigation |
| Browse | 7 | Search, tag filter, clear, recipe detail from browse |
| Profile | 4 | β slider, save, CF progress |
| Backend API | 8 | Ranked endpoint contracts, CF strategy field, vision mock |
| Stale user recovery | 2 | Deleted user redirect to onboarding |
| Shopping List | 11 | Add missing, navigate to list, check off, clear purchased, deduplication |

### Demo recording

```bash
cd frontend
npx playwright test e2e/demo.spec.ts --config=playwright-demo.config.ts
```

Produces `demo-video/demo-eXpairing-full-feature-demo/video.webm`.
Covers all 10 sections: Onboarding → Pantry (with autocomplete demo) → Vision scan → Recipe feed (sort + score breakdown) → Cook & Rate ×5 → CF warm transition → Recipe detail → Browse → Profile → Shopping list.

### Key tests for the grade

```python
# CF personalization works
test_vegan_scores_differ_from_omnivore    # cold start: different users, different scores
test_pantry_changes_scores                # pantry shifts cold-start rankings
test_no_sim_matrix_falls_back_to_preference_scores  # CF never returns zeros

# CF-first routing is correct
test_cf_scores_influence_ranking          # CF always affects output
test_cold_start_used_even_with_mf_...   # new user → cold start, even if MF model loaded
test_transition_at_exact_threshold        # 5th rating → is_warm True

# β learning converges
test_converges_to_zero_waste              # 30 cycles zero-missing → β → 0.95
test_converges_to_permissive             # 30 cycles 4-missing → β → 0.05

# Ranking makes semantic sense
test_french_toast_beats_carbonara         # expiring milk+eggs → french toast first
test_lobster_bisque_is_last              # 0 pantry overlap → dead last

# Live behavioral correctness
test_is_warm_flips_at_exactly_5_ratings  # integration: precise threshold
test_near_expiry_produces_nonzero_urgency # integration: urgency math end-to-end
test_high_beta_ranks_pantry_matches_higher # integration: beta affects real rankings
```

---

## 18. Configuration Reference

### Backend environment variables

| Variable | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./fridge2fork.db` | DB connection |
| `OPENAI_API_KEY` | — | Required for real vision scan |
| `VISION_MOCK` | `false` | Force demo scan mode |
| `TEST_API_BASE` | `http://localhost:8000` | Base URL for integration tests |

### ML tuning parameters

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `DEFAULT_HALF_LIFE_DAYS` | `expiry.py` | 3.0 | Days for urgency to halve |
| `FUZZY_THRESHOLD` | `ingredient_match.py` | 75 | Min rapidfuzz score |
| `MIN_RATINGS_FOR_CF` | `serve_cf.py` | 5 | Ratings to activate biased MF |
| `N_SEEDS` | `cold_start.py` | 30 | Cold-start anchor recipes |
| `PANTRY_SEED_WEIGHT` | `cold_start.py` | 0.4 | Pantry vs tag weight in fallback |
| `LEARNING_RATE` | `beta_updater.py` | 0.15 | Daily β drift speed |
| `LOOKBACK_DAYS` | `beta_updater.py` | 30 | Event window for β update |
| `MIN_EVENTS` | `beta_updater.py` | 3 | Min cook events to trigger |
| `TOP_K_SIMILAR` | `item_similarity.py` | 50 | Neighbors per recipe |
| `MIN_RATINGS` | `item_similarity.py` | 5 | Min ratings to include recipe in matrix |
| `CHUNK_SIZE` | `item_similarity.py` | 500 | Rows per cosine batch |
| `n_factors` | `train_cf.py` | 50 | MF latent dimensions |
| `RELEVANT_THRESHOLD` | `evaluate.py` | 4.0 | Stars = "relevant" |

---

## 19. Demo Script

### Before the demo

```bash
rm -f fridge2fork.db
python3 -m backend.db.seed_dev
python3 -m backend.ml.train_cb       # optional, ~2s
python3 -m uvicorn backend.main:app --port 8000 &
cd frontend && npm run dev
```

### Walkthrough (~12 minutes)

**1. Onboarding (1 min)**
Open http://localhost:5173. Name + β slider + "vegetarian" tag → "Get started →"

> "The beta slider sets your starting waste-aversion. The system learns your true preference from what you actually cook."

**2. Pantry (2 min)**
Add items manually. Type "but" in the ingredient field — watch the autocomplete dropdown suggest "butter" from the Food.com corpus. Arrow down + Enter to select. Fill expiry dates.

Click "Demo scan." Show the vision flow: raw product names → canonical ingredients → expiry inputs → confirm. Pantry refreshes.

> "Autocomplete constrains ingredients to the same vocabulary used in recipes, preventing egg vs eggs mismatches. In production the scan uses GPT-4o vision — demo mode shows the same UX without an API key."

**3. Recipe feed (3 min)**
Click "Recipes." Show the blue cold-start banner and the subtitle: *ranked by collaborative filtering (CF) · expiry urgency · pantry match*.

Click "▼ Why this recipe?" on the top card. Walk through the 4 score component bars:
- **Expiry urgency** — milk expires tomorrow
- **Ingredient match** — you have most things
- **Community score (CF)** — community signal, cold start mode
- **Profile match (CB)** — ingredient TF-IDF cosine match

> "CF strategy: item_based_cold_start. CF has the highest weight (γ=0.35) — taste comes first."

Show the **Sort by** dropdown. Switch to "Expiry urgency" — cards reorder to surface most-urgent recipes. Switch to "CF score" — pure collaborative filtering order. Return to "Total score."

Cook the top recipe. Star rating appears. Give 4 stars. Card turns green.
Click the recipe name to open the full detail page — ingredients + numbered steps from Food.com.

**4. Rate 5 recipes — CF warm transition**
Return to feed. Cook and rate 4 more recipes. Click Refresh — banner flips from blue (cold start) to green ("Personalized from your history").

**5. Shopping list (1 min)**
On a recipe card, click "＋ Buy missing." Button changes to "Added to list." Navigate to the **List** tab. Show items with source recipe attribution. Check off two items. Click "Clear purchased."

> "The shopping list is persistent — items survive navigation and page reloads."

**6. Profile (1 min)**
Click "Profile." Show rating progress: 5/5 — full personalization active. Drag β toward "Use what I have." Save. Return to feed — ranking shifts toward high-pantry-match recipes.

**7. Evaluation (1 min)**
In terminal:
```bash
python3 -m backend.ml.evaluate
```

Show output: RMSE improvement over baseline, Precision@K, ablation showing CF > CB > domain-only.

> "CF provides the strongest signal. The hybrid improves over any single component."

**8. Beta updater (30 sec)**
```bash
python3 -m backend.services.beta_updater --dry-run
```

Show: user cooked 0-missing recipes → revealed_beta=1.0 → beta drifts up.

> "Stated preference vs revealed preference gap. The system closes it automatically."

**9. Tests (30 sec)**
```bash
python3 -m pytest tests/test_ml_behavior.py -v --tb=short
```

Show 38 behavioral tests passing — CF cold/warm transition, pantry effects, vision confirm, score math.

---

---

## 20. Recent Features and Improvements

### Ingredient Autocomplete

**Problem:** Free-text pantry input allowed "egg" when the corpus uses "eggs", breaking ingredient match scoring.

**Solution:** `GET /pantry/suggest?q=<prefix>&limit=8` returns canonical ingredient names from the recipe corpus vocabulary. Prefix matches are returned before substring matches. The vocabulary is lazily built on first call by tokenizing all `ingredients_csv` values in the Recipe table.

**Frontend:** `IngredientAutocomplete.tsx` debounces requests at 180ms and uses a `useRef` (not `useState`) to track input focus in async callbacks — necessary because `useState` captures a stale value in the async closure, while `ref.current` always reflects live state. The dropdown only opens if the input is still focused when the API response arrives, preventing the dropdown from blocking the Add button.

### Shopping List

**Backend:** New `ShoppingListItem` ORM model and `shopping.py` router. `POST /shopping/{user_id}` deduplicates by ingredient name per user — returns `{added: [...], skipped: [...]}` so the UI can display feedback. Items store optional source recipe attribution. `DELETE /shopping/{user_id}?checked_only=true` clears only purchased items.

**Frontend:** `ShoppingListPage.tsx` at `/list` (nav link "List"). `RecipeCard` gains a "＋ Buy missing" button that posts the card's `missing_ingredients` list with recipe attribution.

**Tests:** `test_shopping.py` — 20 unit tests using `StaticPool` for in-memory SQLite isolation. SQLAlchemy's default in-memory engine creates a fresh database per connection; `StaticPool` forces all connections (fixtures and TestClient requests) to share one database.

### Feed Sort Controls

**Frontend only — no backend change.** `RecipeFeedPage` accepts a sort key via a `<select>` dropdown. When the key is `final_score` the server-ordered array is returned as-is. Any other key triggers a client-side `[...visible].sort((a, b) => b[key] - a[key])`. Refreshing the feed resets the sort to `final_score`.

**Subtitle change:** Updated from *"ranked for you"* to *"ranked by collaborative filtering (CF) · expiry urgency · pantry match"* to make CF's primacy explicit.

### UI Polish

Targeted CSS-only changes to move the app away from the archetypal AI-prototype look:
- **Badges:** `border-radius: 9999px` (pill) → `3px` (tag chip) — the single most recognizable AI-prototype signal
- **Cards:** `border-radius: .75rem` → `6px`
- **Buttons and inputs:** `border-radius: .5rem` → `4px`
- **Nav active state:** colored bottom border → black underline (standard product pattern)
- **Nav:** removed drop shadow; reduced height from 56px to 52px
- **Page titles:** reduced size and weight for a less inflated hierarchy

### Score Calibration + MMR Diversity

**Problem:** CF scores on sparse data cluster in a narrow band (e.g., 0.43–0.61). Expiry urgency spans a much wider range. The weighted blend was effectively dominated by expiry, not CF as intended.

**Solution:** `_calibrate(values)` in `scoring.py` — min-max normalizes each component across all candidates before blending. Every component is equally scaled going into the weighted sum.

**Problem:** Top-20 feed filled with near-identical recipes (same cuisine, same main ingredient) because the ranking is greedy relevance-only.

**Solution:** `_mmr_rerank(candidates, top_n, lambda_=0.7)` — selects 20 from the top-60 candidates using Maximal Marginal Relevance. Ingredient Jaccard similarity is the diversity penalty. The first selected recipe is always the highest-scored; subsequent picks balance relevance vs. ingredient dissimilarity to already-selected recipes.

### Implicit Feedback Augmentation (train_cf.py)

**Problem:** Many users cook frequently but rate rarely. Biased MF has no training signal for them and stays in cold-start mode indefinitely.

**Solution:** Cook events are converted to synthetic ratings and merged into matrix factorization training data when no explicit rating exists for that (user, recipe) pair:
```
implicit_rating = max(3.0, 4.0 − min(n_missing, 3) × 0.3)
```
Explicit ratings always take precedence. Cook events are deduplicated: multiple cooks of the same recipe → highest synthetic rating. Net effect: users who cook 5 recipes before rating any reach warm CF sooner.

CLI: `python3 -m backend.ml.train_cf --no-implicit` to train on explicit ratings only (for comparison).

### CB Taste Profile for Warm Users (serve_cb.py)

**Problem:** Cold-start CB uses pantry as a proxy for taste. For warm users who have rated recipes, pantry content is a weak signal — taste is better captured from rating history.

**Solution:** `cb_taste_profile_batch(rated_recipe_ids, ratings, candidate_recipe_ids)` builds a weighted-average TF-IDF profile from rated recipes (weight = rating − 3.0). Positive weight: liked recipes pull the profile toward their ingredients. Negative weight: disliked recipes push the profile away. Negative cosine similarities are clipped to 0 (no active penalty). The recipes.py router selects this path automatically for users with ≥ 5 ratings.

### 7-Day Skip Exclusion (recipes.py)

**Problem:** Skipped recipes reappear immediately on the next feed refresh, frustrating users.

**Solution:** Before scoring, recipe IDs the user has skipped in the last 7 days are excluded from the candidate pool. The cutoff is a datetime comparison on `UserEvent.created_at`.

### Revealed Beta Visibility (users.py + ProfilePage.tsx)

**Problem:** Users set an aspirational β (slider) but their cooking behavior reveals a different true preference. This gap was computed by `beta_updater.py` but never surfaced to the user.

**Solution:**
- `GET /users/{id}/stats` now includes `revealed_beta` and `avg_missing` (computed from cook events with `_compute_revealed_beta` from beta_updater.py)
- Profile page shows an amber warning when `|revealed_beta − stated_beta| > 0.1`:
  *"Your cooking history suggests 78% availability focus — your slider is set to 50%."*

### NDCG@K + Lifecycle Simulation + Weight Grid Search (evaluate.py)

Three new evaluation capabilities:

**NDCG@K** (`evaluate_ndcg`): graded ranking metric that rewards putting highly-rated items at rank 1, not just in the top-K set.

**Lifecycle simulation** (`lifecycle_simulation`): simulates a user from 0 → N ratings, measuring NDCG@10 at each step. Validates the soft blend ramps smoothly.

**Weight grid search** (`tune_weights`): grid over (γ, α) to find the combination with highest NDCG@10. If the best config beats defaults by > 0.005, the output recommends updating `DEFAULT_GAMMA` / `DEFAULT_ALPHA` in scoring.py.

---

*eXpairing — Recommender Systems Workshop, Tel Aviv University*
