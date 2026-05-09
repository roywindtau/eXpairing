"""
train_cb.py
-----------
Builds TF-IDF ingredient embeddings for every recipe in the DB,
then saves the embedding matrix and vocabulary to disk.

Content-based similarity works like this:
    - Each recipe is represented as a TF-IDF vector over its ingredients.
    - At serving time, the user's current pantry is also vectorized.
    - Cosine similarity between the pantry vector and each recipe vector
      gives the CB score: recipes whose ingredient profiles best match
      the pantry's ingredient profile score highest.

This captures cuisine affinity (a pantry with miso, soy sauce, and
sesame oil will naturally cosine-match Japanese recipes more).

Saved artifacts:
    models/cb_matrix.npz       -- sparse TF-IDF matrix (n_recipes x vocab)
    models/cb_recipe_ids.npy   -- recipe_id for each row in the matrix
    models/cb_vectorizer.pkl   -- fitted TfidfVectorizer (needed to transform
                                   new pantry vectors at serve time)
    models/cb_meta.json        -- training stats

Run:
    python -m backend.ml.train_cb
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import Recipe

MODELS_DIR      = Path("models")
CB_MATRIX       = MODELS_DIR / "cb_matrix.npz"
CB_RECIPE_IDS   = MODELS_DIR / "cb_recipe_ids.npy"
CB_VECTORIZER   = MODELS_DIR / "cb_vectorizer.pkl"
CB_META         = MODELS_DIR / "cb_meta.json"


def load_recipes() -> tuple[list[int], list[str]]:
    """
    Load all recipes from DB.
    Returns (recipe_ids, ingredient_documents).
    Each document is the recipe's ingredients joined by space —
    the unit TF-IDF operates on.
    """
    print("Loading recipes from DB ...")
    db = SessionLocal()
    try:
        recipes = db.query(Recipe.id, Recipe.ingredients_csv).all()
    finally:
        db.close()

    recipe_ids = []
    documents  = []
    for r_id, ingredients_csv in recipes:
        # Each comma-separated ingredient becomes space-separated words in doc.
        # "eggs,whole milk,black pepper" -> "eggs whole milk black pepper"
        ingredients = [ing.strip() for ing in ingredients_csv.split(",") if ing.strip()]
        doc = " ".join(ingredients)
        if doc:
            recipe_ids.append(r_id)
            documents.append(doc)

    print(f"  Loaded {len(recipe_ids):,} recipes.")
    return recipe_ids, documents


def train() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Load recipes
    recipe_ids, documents = load_recipes()
    if not recipe_ids:
        print("No recipes found. Run seed_recipes.py first.")
        sys.exit(1)

    # 2. Fit TF-IDF
    # ngram_range=(1,2) captures both single ingredients ("garlic") and
    # bigrams ("garlic powder") as separate features.
    # min_df=2 drops hapax ingredients that appear in only one recipe
    # (they add noise without helping similarity).
    # min_df: drop terms that appear in fewer than this many docs.
    # With a small dev corpus (20 recipes) use 1; with the full dataset use 2.
    min_df = 1 if len(documents) < 100 else 2

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=20_000,   # cap vocab size for memory
        sublinear_tf=True,     # log(1+tf) instead of raw tf
        analyzer="word",
        token_pattern=r"[a-z]+",  # single words; ngrams capture multi-word ingredients
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    vocab_size   = len(vectorizer.vocabulary_)
    print(f"  Matrix shape: {tfidf_matrix.shape}  "
          f"(recipes x vocab)  |  vocab size: {vocab_size:,}")

    # 3. Save artifacts
    sp.save_npz(CB_MATRIX, tfidf_matrix)
    np.save(CB_RECIPE_IDS, np.array(recipe_ids, dtype=np.int32))

    with open(CB_VECTORIZER, "wb") as f:
        pickle.dump(vectorizer, f)

    meta = {
        "trained_at":  datetime.now().isoformat(),
        "n_recipes":   len(recipe_ids),
        "vocab_size":  vocab_size,
        "matrix_shape": list(tfidf_matrix.shape),
        "ngram_range": [1, 2],
        "min_df":      2,
        "max_features": 20_000,
    }
    with open(CB_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {CB_MATRIX}")
    print(f"  Saved -> {CB_RECIPE_IDS}")
    print(f"  Saved -> {CB_VECTORIZER}")
    print(f"  Saved -> {CB_META}")
    print(f"\nDone. {len(recipe_ids):,} recipes embedded into {vocab_size:,}-dim TF-IDF space.")


if __name__ == "__main__":
    train()
