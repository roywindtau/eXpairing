# Installation Guide

This guide covers setup instructions for running exPairing locally, executing full ML training pipelines for both Recipe and Drinks/Wine recommender models, configuring AI vision provider credentials, running Docker containers, and executing the automated test suites.

## Prerequisites

| Tool | Version | Verification Command | Notes |
|---|---|---|---|
| **Python** | 3.9 or later | `python3 --version` | Standard macOS/Linux system Python 3.9 work. All commands use `python3`. |
| **pip** | Any compatible | `pip3 --version` | Python package manager. |
| **Node.js** | 18 or later | `node --version` | Required for the frontend Vite application. |
| **npm** | 9 or later | `npm --version` | Node package manager. |
| **Docker & Compose** | Desktop / Engine | `docker compose version` | Optional. Recommended for quick containerized environment setup. |
| **Kaggle Credentials** | API Token | — | `kaggle.json` required only when executing full offline training on Food.com / X-Wines datasets. |
| **AI Vision API Token** | OpenAI / Gemini | — | Optional API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) for live multi-modal fridge photo scanning. Built-in mock scanner runs automatically if omitted. |

## Installation Steps

All commands should be executed from the **project root directory** unless explicitly noted otherwise.

### Step 1 — Clone Repository & Setup Virtual Environment
```bash
git clone <repository-url>
cd recsys26

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required Python dependencies
pip3 install -r requirements.txt
```

### Step 2 — Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 3 — Seed Database & Model Artifacts (Choose Option A or Option B)

#### Option A: Quick Start (Local Demo, No Kaggle API needed)
Seeds the SQLite database (`fridge2fork.db`) with 20 diverse dev recipes, a demo pantry, and a 100-wine demo catalog.
```bash
# Seed dev recipe database
python3 -m backend.db.seed_dev

# Create wine tables and seed the committed 100-wine sample catalog
# (data/wine/clean_wines.sample.csv — ships with popularity stats baked in,
# so no dataset download and no stats computation is needed)
python3 -m backend.db.reset_wines
python3 -m backend.db.wine.seed_wines
```
*Note*: `./data/wine/clean_and_seed.sh` and `compute_wine_stats` require the raw X-Wines download — they belong to Option B, not the quick start.

#### Option B: Full Dataset Pipeline & Offline ML Training
For training production models on 231k recipes (Food.com) and 100k wines / 21M ratings (X-Wines):
1. Download `kaggle.json` from your Kaggle account (Settings → API Token).
2. Place `kaggle.json` at `~/.kaggle/kaggle.json` (or in the project root directory).
3. Execute the full offline training pipeline:

```bash
# --- 1. Recipe Pipeline Training ---
python3 -m data.download_foodcom                  # Downloads RAW_recipes.csv & RAW_interactions.csv
python3 -m backend.db.seed_recipes                # Seeds 231k recipes into SQLite DB
python3 -m backend.db.seed_ratings                # Seeds 1.1M explicit ratings into DB
python3 -m backend.ml.item_similarity             # Generates models/item_sim_matrix.npz (top-50 sparse item-item CF)
python3 -m backend.ml.train_cf                    # Trains models/cf_model.pkl (Biased Funk SVD via SGD)
python3 -m backend.ml.train_cb                    # Trains models/cb_matrix.npz (TF-IDF ingredient vectors)
python3 -m backend.ml.evaluate                    # Evaluates RMSE, Precision@K, NDCG@K → models/eval_results.json

# --- 2. Drinks & Wine Pipeline Training ---
python3 -m data.wine.download_wines               # Downloads X-Wines raw corpus
python3 -m data.wine.clean_wines                  # Generates clean_wines.csv & clean_ratings.csv
python3 -m backend.db.reset_wines                 # Recreates wine tables
python3 -m backend.db.wine.seed_wines             # Seeds the full 100k-wine catalog into the DB (training scripts below read wines from the DB)
python3 -m backend.db.wine.compute_wine_stats     # Aggregates 21M ratings → popularity prior
python3 -m data.wine.region_rollup                # Collapses 2,160 appellations → 107 parent regions (models/region_rollup.json)
python3 -m backend.ml.wine.training.train_cb      # Generates models/wine_cb_matrix.npz (structured content-based matrix)
python3 -m backend.ml.wine.training.build_wine_split # Creates frozen leave-5-out evaluation split in models/wine_split/
python3 -m backend.ml.wine.training.eval_wine_popularity # Popularity ranking baseline (the floor ALS must beat, NDCG@10 ≈ 0.0071)
python3 -m backend.ml.wine.training.train_wine_als   # Trains confidence-weighted ALS model (models/wine_als_model.npz) and evaluates it on the frozen split
python3 -m data.wine.inspect_neighbors            # Diagnostic tool to sanity-check content-based neighbor weights

# --- 3. Recipe-Wine Pairing Pipeline Training ---
python3 -m data.pairing.download_pairing             # Downloads wine_food_pairings.csv (~35k labeled wine/food pairings)
python3 -m data.pairing.extract_pairing_rules        # Reads data/pairing/wine_food_pairings.csv → models/pairing_rules.json (empirical sommelier rules)
python3 -m data.pairing.build_wine_pairing_vectors   # Generates models/wine_pair_matrix.npz + wine_pair_meta.json (12-dim category vectors)
```

### Step 4 — Configure AI Vision Provider Tokens (Optional)
The fridge photo scanning feature uses vision AI to identify items and read expiration dates. You can configure either OpenAI GPT-4o or Google Gemini 2.5 Flash API tokens. Without an API token, the application seamlessly falls back to a built-in mock scanner returning realistic test items.

```bash
# Option 1: OpenAI GPT-4o Vision
export OPENAI_API_KEY="sk-..."

# Option 2: Google Gemini 2.5 Flash Vision
export GEMINI_API_KEY="AIzaSy..."

# Alternatively, define API keys in a .env file at the project root:
echo "OPENAI_API_KEY=sk-..." > .env
```

### Step 5 — Start the Local Servers or Run via Docker

#### Running Locally (Hot-Reloading Dev Servers)
Terminal 1 (Backend API):
```bash
python3 -m uvicorn backend.main:app --reload --port 8000
```
Terminal 2 (Frontend UI):
```bash
cd frontend
npm run dev
```

#### Running via Docker
The recommended way to run the full stack containerized:
```bash
./dev.sh                 # Starts containers; auto-seeds dev DB only if fridge2fork.db is missing (backend: port 8000, frontend: port 5173)
./dev.sh --seed          # Forces a fresh dev DB seed before starting (safe to re-run)
./dev.sh --rebuild       # Forces container rebuild after modifying dependencies
```
*Note on Docker Wiring:* `docker compose` merges `docker-compose.override.yml` over `docker-compose.yml` to run Vite dev server with HMR. To run production static build with nginx instead:
```bash
docker compose -f docker-compose.yml up --build
```

### Step 6 — Deploy to a Hosted Platform (Optional)

Both images run unmodified on a Docker-based PaaS (Render, Railway, Fly.io), so the
application can be served from a URL without a local clone or a training run. The
steps below use Render.

**Prerequisite — publish the trained artifacts.** `models/` (~468MB) and the seeded
`fridge2fork.db` (~474MB) are gitignored and exceed GitHub's 100MB per-file limit, so
the image fetches them at build time instead of copying them from the repository.
Build a tarball from a machine that has completed Option B above, and attach it to a
release on a **public** repository (a private repository's release assets are not
anonymously downloadable, and the build would fail with a 404):

```bash
# From the project root, after Option B training has produced models/ and the DB
tar -czf artifacts.tar.gz --exclude='models/wine_split' models fridge2fork.db
gh release create artifacts-v1 artifacts.tar.gz --title "Trained artifacts"

# The public download URL to pass as ARTIFACTS_URL
gh release view artifacts-v1 --json assets -q '.assets[].url'
```

`models/wine_split/` is excluded because it is the frozen evaluation split, read only
by the training and evaluation scripts and never at serve time.

**Deploy the backend.** Create a Render Web Service from the repository:

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `Dockerfile.backend` |
| Root directory | `.` (repository root) |
| Build argument | `ARTIFACTS_URL` = the release asset URL from above |

The build downloads and unpacks the artifacts, then asserts that `fridge2fork.db`,
`models/cf_model.pkl`, and `models/wine_als_model.npz` are present — a truncated or
wrongly-shaped tarball fails the build rather than producing an image that errors on
its first request. Note the assigned URL (e.g. `https://expairing-api.onrender.com`).

**Deploy the frontend.** Create a second Web Service from the same repository:

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `Dockerfile.frontend` |
| Root directory | `./frontend` |
| Build argument | `VITE_API_URL` = the backend URL from the previous step |

Vite inlines `VITE_API_URL` into the JavaScript bundle at build time, so the backend
must be deployed (or its URL known) first. Changing it later requires a rebuild, not
just a restart.

**Allow the frontend origin.** On the *backend* service, set the environment variable
`CORS_ORIGINS` to the frontend's URL (comma-separated for several). `http://localhost:5173`
and `http://localhost:3000` remain allowed by default, so local development is unaffected.

**Neither image hardcodes a port.** Both read the `$PORT` that hosted platforms inject —
the backend passes it to uvicorn, and the frontend renders it into the nginx config at
container start. No configuration is needed for this.

*Free-tier caveat:* Render's free web services sleep after roughly 15 minutes of
inactivity, so the first request after an idle period can take 30–60 seconds while the
container wakes. Both services sleep independently.

## Post‑install / Verification

* **Backend Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```
  Expected output: `{"status":"ok"}`.
* **Interactive API Documentation**: Access Swagger UI at `http://localhost:8000/docs`.
* **Frontend Web App**: Navigate to `http://localhost:5173` in your browser.
* **Probe Wine API Endpoints Directly**:
  ```bash
  # Fetch top 10 popular/personalized wines
  curl "http://localhost:8000/wine/ranked?top_n=10" | jq

  # Log a wine rating event (1-5 stars)
  curl -X POST "http://localhost:8000/wine-events" \
       -H "Content-Type: application/json" \
       -d '{"user_id":1,"wine_id":100001,"event_type":"rate","rating":4.5}'
  ```

### Running Test Suites
Execute backend and frontend test coverage to verify system correctness:

```bash
# 1. Run all unit + behavioral integration tests (530+ tests)
python3 -m pytest tests/ -v

# 2. Run unit tests only (fast, no live backend needed)
python3 -m pytest tests/ -v --ignore=tests/test_ml_behavior.py

# 3. Run End-to-End Playwright frontend tests (63 tests)
cd frontend
npx playwright test --reporter=list
```

## Troubleshooting

* **Port Conflicts (8000 or 5173 already in use)**: Verify no orphaned `uvicorn` or `vite` processes are running. Kill running background processes on port 8000 or 5173 before launching.
* **Flat Wine Popularity Scores**: If wine recommendations return identical popularity scores after a full-dataset seed (Option B), ensure `python3 -m backend.db.wine.compute_wine_stats` was executed to aggregate ratings into the popularity prior. (The Option A sample catalog ships with popularity stats pre-computed, so this step does not apply there.)
* **Missing AI Vision Tokens**: If neither `OPENAI_API_KEY` nor `GEMINI_API_KEY` is configured, the `/vision/scan` endpoint will raise a runtime error if called directly, but the UI demo scanner invokes `/vision/mock` automatically. Set an API token in `.env` for real photo processing.
* **SQLite Database Lock Errors**: SQLite locks the database file during write operations. Restart uvicorn processes or execute `python3 -m backend.db.seed_dev` to reset state if locks persist.
