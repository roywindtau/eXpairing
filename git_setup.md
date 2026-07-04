# eXpairing — Git & Setup Guide

## What lives in git / what doesn't

### In git (committed)
| Path | Why |
|---|---|
| `backend/` | All Python source — routers, services, ML code, DB models |
| `frontend/src/` | All React/TypeScript source |
| `frontend/{index.html,package.json,package-lock.json,tsconfig.json,vite.config.ts}` | Frontend scaffold & lockfile |
| `frontend/playwright*.config.ts` | E2E config |
| `frontend/e2e/` | Playwright test specs |
| `tests/` | Pytest test suite |
| `data/__init__.py`, `data/download_foodcom.py`, `data/explore_foodcom.ipynb` | Download script & notebook |
| `models/.gitkeep` | Keeps the empty `models/` directory tracked |
| `requirements.txt` | Python dependencies |
| `train_pipeline.sh` | One-shot ML training script |
| `docker-compose.yml`, `Dockerfile.*` | Container config |
| `.gitignore` | This file |
| `README.md`, `EXPAIRING.md`, `algoclass.md` | Documentation |

### NOT in git (regenerated or secret)

| Path | Why not | How to recreate |
|---|---|---|
| `models/*.pkl / *.npz / *.npy / *.json` | Large ML artifacts (CF model ~80MB) | Run `train_pipeline.sh` |
| `data/RAW_recipes.csv` | ~230MB Kaggle dataset | `python -m data.download_foodcom` |
| `data/RAW_interactions.csv` | ~55MB Kaggle dataset | Same as above |
| `fridge2fork.db` | SQLite database — runtime state | `python -m backend.db.seed_dev` (dev) or pipeline (full) |
| `kaggle.json` | API credentials — never commit | Kaggle account → Settings → API token |
| `frontend/node_modules/` | npm packages | `npm install` |
| `frontend/dist/` | Built frontend | `npm run build` |
| `frontend/demo-video/` | Large video (~5MB) | `npx playwright test e2e/demo.spec.ts --config=playwright-demo.config.ts` |
| `frontend/playwright-report/` | Test output | Re-run tests |
| `.DS_Store`, `__pycache__/`, `.pytest_cache/` | OS/tool noise | — |

---

## Setup on a new machine

### Prerequisites

- Python 3.9 (the codebase uses 3.9-compatible syntax; 3.10+ also works)
- Node.js 18+ and npm
- A Kaggle account (only needed for the full dataset; dev mode skips it)

---

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd smartrecipes
```

---

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Choose: dev mode or full pipeline

#### Option A — Dev mode (fast, ~30 seconds, no Kaggle needed)

Seeds a local SQLite database with 20 hand-picked recipes and one demo user.
The ML models won't be loaded, so CF scores will show "not available" — everything
else (pantry, expiry urgency, ingredient match, content-based) works fully.

```bash
python -m backend.db.seed_dev
```

Skip to step 5.

#### Option B — Full pipeline (takes ~30–60 min, needs Kaggle)

This loads 231k recipes, 1.07M ratings, and trains all three ML models.

**a. Kaggle credentials**

Download your API token from Kaggle (Account → Settings → Create New API Token).
It gives you a `kaggle.json` file. Place it in one of two places:

```bash
# Option 1: standard Kaggle location (works everywhere)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Option 2: project root (train_pipeline.sh reads it from here too)
cp /path/to/kaggle.json smartrecipes/kaggle.json
```

**Never commit `kaggle.json` — it's in `.gitignore` for this reason.**

**b. Run the full pipeline**

```bash
chmod +x train_pipeline.sh
./train_pipeline.sh
```

What it does, in order:
1. Downloads `data/RAW_recipes.csv` and `data/RAW_interactions.csv` from Kaggle
2. Seeds 231k recipes into `fridge2fork.db`
3. Seeds 1.07M ratings into the DB
4. Trains item-similarity matrix → `models/item_sim_*.{npz,npy,json}`
5. Trains SVD matrix factorization → `models/cf_model.pkl`, `models/cf_meta.json`
6. Trains TF-IDF content-based model → `models/cb_*.{pkl,npz,npy,json}`
7. Runs offline evaluation and prints metrics

For a quick test with a subset:
```bash
./train_pipeline.sh 10000    # 10k recipes instead of 231k
```

**Note:** The CF model is trained with `--no-implicit` (explicit Food.com star ratings only,
no cook events). See `EXPAIRING.md` for the reasoning.

---

### 4. (Optional) OpenAI key for vision scanning

The fridge photo scan feature calls GPT-4o Vision. Without a key it falls back to a
deterministic mock that returns realistic demo items.

```bash
export OPENAI_API_KEY=sk-...
# or put it in a .env file at the project root:
echo "OPENAI_API_KEY=sk-..." > .env
```

---

### 5. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Verify:
```bash
curl http://localhost:8000/health        # → {"status":"ok"}
curl http://localhost:8000/docs          # Swagger UI in browser
```

---

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in a browser. The app proxies API calls to `localhost:8000`.

---

### 7. Run the test suite

```bash
# from the smartrecipes/ root
python -m pytest tests/ -v
```

Expected: 387 passed, 1 skipped (the skip is a DB-state test that needs the full dataset).

---

### 8. Re-record the demo video (optional)

```bash
cd frontend
npx playwright test e2e/demo.spec.ts --config=playwright-demo.config.ts
# Output: frontend/demo-video/demo-eXpairing-full-feature-demo/video.webm
```

---

## Startup order summary

Every time you come back to the project, just:

```bash
# terminal 1 — backend
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm run dev
```

Both servers must be running for the app to work. The frontend talks to the backend at
`http://localhost:8000` (configured in `frontend/src/api/client.ts`).
