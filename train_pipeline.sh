#!/usr/bin/env bash
# train_pipeline.sh
# -----------------
# Full ML training pipeline for Fridge2Fork.
# Run this once after downloading the Food.com dataset.
#
# Steps:
#   1. Download Food.com dataset from Kaggle
#   2. Seed recipes into the DB
#   3. Seed ratings (user interactions) into the DB
#   4. Train item-similarity matrix (item-based CF, cold start)
#   5. Train SVD matrix factorization (warm CF)
#   6. Train TF-IDF content-based embeddings
#
# Usage:
#   chmod +x data/train_pipeline.sh
#   ./data/train_pipeline.sh
#
# For a quick dev run without the full 230k recipes:
#   ./data/train_pipeline.sh --limit 10000
#
# Prerequisites:
#   pip install -r requirements.txt
#   ~/.kaggle/kaggle.json with your API credentials

set -e   # stop on first error

LIMIT=${1:-0}   # 0 = all rows

echo "========================================"
echo " Fridge2Fork ML training pipeline"
echo "========================================"
echo ""

# ── step 0: download dataset ─────────────────────────────────────────────

if [ ! -f "data/RAW_recipes.csv" ]; then
    echo "[1/6] Downloading Food.com dataset from Kaggle..."
    python -m data.download_foodcom
else
    echo "[1/6] Dataset already present — skipping download."
fi
echo ""

# ── step 1: seed recipes ──────────────────────────────────────────────────

echo "[2/6] Seeding recipes into DB..."
if [ "$LIMIT" -gt 0 ]; then
    python -m backend.db.seed_recipes --limit "$LIMIT"
else
    python -m backend.db.seed_recipes
fi
echo ""

# ── step 2: seed ratings ──────────────────────────────────────────────────

echo "[3/6] Seeding ratings into DB..."
if [ "$LIMIT" -gt 0 ]; then
    python -m backend.db.seed_ratings --limit "$((LIMIT * 5))"
else
    python -m backend.db.seed_ratings
fi
echo ""

# ── step 3: item similarity matrix ────────────────────────────────────────

echo "[4/6] Training item-similarity matrix (item-based CF)..."
python -m backend.ml.item_similarity
echo ""

# ── step 4: SVD collaborative filtering ──────────────────────────────────

echo "[5/6] Training SVD collaborative filtering model..."
python -m backend.ml.train_cf --no-implicit
echo ""

# ── step 5: content-based embeddings ─────────────────────────────────────

echo "[6/6] Training TF-IDF content-based embeddings..."
python -m backend.ml.train_cb
echo ""

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

# ── step 6: evaluation ────────────────────────────────────────────────────

echo "[7/7] Running offline evaluation..."
python -m backend.ml.evaluate --full
echo ""
