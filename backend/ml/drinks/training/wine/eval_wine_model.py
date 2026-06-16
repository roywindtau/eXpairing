"""
eval_wine_model.py
------------------
The ONE evaluation harness for wine ranking. Loads the frozen leave-k-out
split (from build_wine_split.py) and scores a model against it with implicit's
ranking_metrics_at_k. Every wine experiment routes through here so all numbers
are directly comparable.

Provides:
    evaluate(model, k=10) -> dict   # metrics for an implicit model object
    print_metrics(name, metrics)    # consistent reporting

Both train_wine_als.py and the popularity baseline call evaluate() against the
SAME frozen train/test, so a tuning change that moves NDCG is a real change,
not split noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from implicit.evaluation import ranking_metrics_at_k

from backend.ml.drinks.training.wine.build_wine_split import load_split


def evaluate(model, k: int = 10) -> dict:
    """Score an implicit model against the frozen split. Returns metric dict."""
    train, test, _user_ids, _item_ids = load_split()
    metrics = ranking_metrics_at_k(model, train, test, K=k, show_progress=True)
    return {
        "precision": float(metrics["precision"]),
        "map":       float(metrics["map"]),
        "ndcg":      float(metrics["ndcg"]),
        "auc":       float(metrics["auc"]),
        "k":         k,
    }


def print_metrics(name: str, metrics: dict) -> None:
    k = metrics["k"]
    print(f"\n=== {name} (leave-k-out frozen split, @{k}) ===")
    print(f"  precision@{k}: {metrics['precision']:.4f}")
    print(f"  map@{k}:       {metrics['map']:.4f}")
    print(f"  ndcg@{k}:      {metrics['ndcg']:.4f}")
    print(f"  auc:          {metrics['auc']:.4f}")
