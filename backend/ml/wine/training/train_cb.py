"""
train_cb.py
-----------------
TF-IDF content-based embeddings for every wine in the DB.
Mirrors backend/ml/train_cb.py one-to-one but for the Drink table.

The vocabulary (style words, harmonize categories, wine body terms) is what
`flavor_bridge.py` is deliberately designed to inject into recipe docs at
query time, so the cosine in serve_cb compares wines against any recipe
in one shot.

Per-drink documents
-------------------
Wine: "wine {style} {grapes_csv} {harmonize_csv}"
                                                 e.g. "wine red malbec beef lamb grilled"

Saved artifacts
---------------
    models/drink_cb_matrix.npz       sparse TF-IDF matrix (n_drinks x vocab)
    models/drink_cb_ids.npy          drink_id for each row in the matrix
    models/drink_cb_kinds.npy        "wine" for each row (object array)
    models/drink_cb_vectorizer.pkl   fitted TfidfVectorizer
    models/drink_cb_meta.json        training stats

Run:
    python -m backend.ml.wine.training.train_cb
"""

from __future__ import annotations

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
from backend.db.models import Wine

MODELS_DIR        = Path("models")
CB_MATRIX         = MODELS_DIR / "drink_cb_matrix.npz"
CB_IDS            = MODELS_DIR / "drink_cb_ids.npy"
CB_KINDS          = MODELS_DIR / "drink_cb_kinds.npy"
CB_VECTORIZER     = MODELS_DIR / "drink_cb_vectorizer.pkl"
CB_META           = MODELS_DIR / "drink_cb_meta.json"


def _wine_doc(d: Wine) -> str:
    """Compose the per-wine text document fed into TF-IDF."""
    parts: list[str] = ["wine"]
    if d.style:
        parts.append(d.style)
    if d.grapes_csv:
        parts.append(d.grapes_csv.replace(",", " "))
    if d.harmonize_csv:
        parts.append(d.harmonize_csv.replace(",", " "))
    # Review tokens for wine were computed at seed-time from harmonize/grapes/name,
    # so they mostly overlap with the above. Including them is still cheap and
    # gives a small TF boost to repeated tokens.
    if d.review_tokens_csv:
        parts.append(d.review_tokens_csv.replace(",", " "))
    # Lowercase the whole thing so the vectorizer's token_pattern matches.
    return " ".join(parts).lower()


def load_wines() -> tuple[list[int], list[str], list[str]]:
    """Load all wines from DB; return (ids, kinds, documents) with no empties.

    `kinds` is a constant "wine" array, kept only so the serve-time CB loader's
    artifact-existence gate (drink_cb_kinds.npy) stays satisfied.
    """
    print("Loading wines from DB ...")
    db = SessionLocal()
    ids:   list[int] = []
    kinds: list[str] = []
    docs:  list[str] = []
    try:
        for d in db.query(Wine).all():
            doc = _wine_doc(d)
            if not doc.strip():
                continue
            ids.append(d.id)
            kinds.append("wine")
            docs.append(doc)
    finally:
        db.close()

    print(f"  Loaded {len(ids):,} wines.")
    return ids, kinds, docs


def train() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    ids, kinds, documents = load_wines()
    if not ids:
        print("No wines found. Run `python -m backend.db.wine.seed_wines` first.")
        sys.exit(1)

    # Match recipe CB conventions for parity: same ngram range, same
    # token_pattern, same sublinear_tf. Lower min_df because our drink
    # corpus is much smaller than the recipe corpus.
    min_df = 1 if len(documents) < 500 else 2

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=20_000,
        sublinear_tf=True,
        analyzer="word",
        token_pattern=r"[a-z]+",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Matrix shape: {tfidf_matrix.shape}  (drinks x vocab)  |  vocab: {vocab_size:,}")

    sp.save_npz(CB_MATRIX, tfidf_matrix)
    np.save(CB_IDS,   np.array(ids,   dtype=np.int64))
    np.save(CB_KINDS, np.array(kinds, dtype=object))

    with open(CB_VECTORIZER, "wb") as f:
        pickle.dump(vectorizer, f)

    meta = {
        "trained_at":   datetime.now().isoformat(),
        "n_drinks":     len(ids),
        "n_wines":      sum(1 for k in kinds if k == "wine"),
        "vocab_size":   vocab_size,
        "matrix_shape": list(tfidf_matrix.shape),
        "ngram_range":  [1, 2],
        "min_df":       min_df,
        "max_features": 20_000,
    }
    with open(CB_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {CB_MATRIX}")
    print(f"  Saved -> {CB_IDS}")
    print(f"  Saved -> {CB_KINDS}")
    print(f"  Saved -> {CB_VECTORIZER}")
    print(f"  Saved -> {CB_META}")
    print(f"\nDone. {len(ids):,} drinks embedded into {vocab_size:,}-dim TF-IDF space.")


if __name__ == "__main__":
    train()
