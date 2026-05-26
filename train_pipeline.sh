#!/usr/bin/env bash
# train_pipeline.sh
# -----------------
# Full ML training pipeline for Fridge2Fork (recipes + drinks).
#
# Recipe stages:
#   1. Download Food.com dataset from Kaggle
#   2. Seed recipes into the DB
#   3. Seed ratings (user interactions) into the DB
#   4. Train item-similarity matrix (item-based CF, cold start)
#   5. Train SVD matrix factorization (warm CF)
#   6. Train TF-IDF content-based embeddings
#   7. Offline evaluation
#
# Drink stages (Path A pairing + Path B "For You"):
#   D1. Download drink datasets (Beer Reviews from Kaggle + X-Wines Test)
#   D2. Seed Drink table (beers aggregated, wines as-is)
#   D3. Seed DrinkEvent table (beer + wine ratings as external users)
#   D4. Train drink CB (TF-IDF over drink descriptors + bridged tokens)
#   D5. Train drink CF SVD (beers only — wines are too sparse)
#   D6. Build drink item-similarity matrices (one per kind)
#
# Usage:
#   chmod +x train_pipeline.sh
#   ./train_pipeline.sh                 # full pipeline (recipes + drinks)
#   ./train_pipeline.sh --skip-drinks   # recipes only (original behavior)
#   ./train_pipeline.sh --drinks-only   # skip the recipe stages
#   ./train_pipeline.sh 10000           # 10k recipes for quick dev (still does drinks)
#
# Prerequisites:
#   pip install -r requirements.txt
#   ~/.kaggle/kaggle.json with your API credentials (for both Food.com and Beer Reviews)

set -e   # stop on first error

# ── arg parsing ──────────────────────────────────────────────────────────
LIMIT=0
SKIP_DRINKS=0
DRINKS_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --skip-drinks) SKIP_DRINKS=1 ;;
        --drinks-only) DRINKS_ONLY=1 ;;
        *) LIMIT="$arg" ;;
    esac
done

echo "========================================"
echo " Fridge2Fork ML training pipeline"
echo "========================================"
echo ""

# ── recipe stages ────────────────────────────────────────────────────────

if [ "$DRINKS_ONLY" -eq 0 ]; then

    if [ ! -f "data/RAW_recipes.csv" ]; then
        echo "[1/7] Downloading Food.com dataset from Kaggle..."
        python -m data.download_foodcom
    else
        echo "[1/7] Food.com dataset already present — skipping download."
    fi
    echo ""

    echo "[2/7] Seeding recipes into DB..."
    if [ "$LIMIT" -gt 0 ]; then
        python -m backend.db.seed_recipes --limit "$LIMIT"
    else
        python -m backend.db.seed_recipes
    fi
    echo ""

    echo "[3/7] Seeding ratings into DB..."
    if [ "$LIMIT" -gt 0 ]; then
        python -m backend.db.seed_ratings --limit "$((LIMIT * 5))"
    else
        python -m backend.db.seed_ratings
    fi
    echo ""

    echo "[4/7] Training item-similarity matrix (item-based CF)..."
    python -m backend.ml.item_similarity
    echo ""

    echo "[5/7] Training SVD collaborative filtering model..."
    python -m backend.ml.train_cf --no-implicit
    echo ""

    echo "[6/7] Training TF-IDF content-based embeddings..."
    python -m backend.ml.train_cb
    echo ""

    echo "[7/7] Running offline evaluation..."
    python -m backend.ml.evaluate --full
    echo ""

else
    echo "  --drinks-only set: skipping recipe stages."
    echo ""
fi

# ── drink stages ─────────────────────────────────────────────────────────

if [ "$SKIP_DRINKS" -eq 0 ]; then

    echo "========================================"
    echo " Drink recommender stages"
    echo "========================================"
    echo ""

    if [ ! -f "data/beer_reviews.csv" ]; then
        echo "[D1/6] Downloading beer dataset (Beer Reviews)..."
        python -m data.drinks.download_beer
    else
        echo "[D1/6] Beer dataset already present — skipping download."
    fi
    echo ""

    echo "[D2/6] Seeding Drink table (beers + wines)..."
    python -m backend.db.drinks.seed_drinks
    echo ""

    echo "[D3/6] Seeding DrinkEvent table (beer + wine ratings)..."
    python -m backend.db.drinks.seed_ratings
    echo ""

    echo "[D4/6] Training drink CB (TF-IDF + flavor bridge)..."
    python -m backend.ml.drinks.training.train_cb
    echo ""

    echo "[D5/6] Training drink CF SVD (beers only — wines too sparse)..."
    python -m backend.ml.drinks.training.train_cf
    echo ""

    echo "[D6/6] Building drink item-similarity matrices..."
    python -m backend.ml.drinks.training.item_similarity
    echo ""

else
    echo "  --skip-drinks set: skipping drink stages."
    echo ""
fi

# ── done ──────────────────────────────────────────────────────────────────

echo "========================================"
echo " Training complete!"
echo ""
echo " Artifacts saved to models/:"
ls -lh models/ 2>/dev/null || echo "  (models/ not found)"
echo ""
echo " Start the server:"
echo "   uvicorn backend.main:app --reload"
echo "========================================"
