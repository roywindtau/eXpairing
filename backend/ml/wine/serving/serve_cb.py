"""
serve_cb.py
-----------
Content-based wine scoring at request time.

Loads the precomputed structured wine matrix (wine_cb_matrix.npz, UNWEIGHTED),
applies the sommelier weights per block at load, and scores candidate wines
against a user's taste profile.

Taste profile = rating-weighted average of the user's liked wines' vectors
(weight = rating - 3.0, like the recipe taste-profile CB). A user who rated a
wine 5 pulls hard toward it; a 2 pushes away.

Weights are applied here (not baked into the artifact) so they can be retuned —
or overridden per request ("emphasize grape") — without recomputing the matrix.

Public API
----------
    cb_available() -> bool
    cb_scores(liked: list[(wine_id, rating)], candidate_ids) -> dict[wine_id, score]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

MODELS_DIR = Path("models")
_MATRIX = MODELS_DIR / "wine_cb_matrix.npz"
_IDS    = MODELS_DIR / "wine_cb_ids.npy"
_BLOCKS = MODELS_DIR / "wine_cb_blocks.json"

# Sommelier weights (normalized). Default; callers may override per request.
DEFAULT_WEIGHTS = {"acidity": 0.368, "body": 0.368, "region": 0.158,
                   "abv": 0.053, "grape": 0.053}

_state: dict | None = None


def _load() -> dict | None:
    """Lazy-load artifacts once. Returns None if CB hasn't been built."""
    global _state
    if _state is not None:
        return _state or None
    if not (_MATRIX.exists() and _IDS.exists() and _BLOCKS.exists()):
        _state = {}
        return None
    mat = sp.load_npz(_MATRIX).tocsr().astype(np.float64)
    ids = np.load(_IDS)
    blocks = json.loads(_BLOCKS.read_text(encoding="utf-8"))
    _state = {
        "mat": mat,
        "ids": ids,
        "row_of": {int(w): i for i, w in enumerate(ids)},
        "blocks": blocks["blocks"],
        "grape_vocab": blocks.get("grape_vocab", {}),
        "dim": int(blocks.get("dim", mat.shape[1])),
    }
    return _state


def _apply_weights(mat: sp.csr_matrix, blocks: dict, weights: dict) -> sp.csr_matrix:
    """Scale each block of a (copy of the) matrix by its weight."""
    m = mat.tolil(copy=True)
    g, r = blocks["grape"], blocks["region"]
    m[:, g["start"]:g["end"]] = m[:, g["start"]:g["end"]] * weights["grape"]
    m[:, r["start"]:r["end"]] = m[:, r["start"]:r["end"]] * weights["region"]
    m[:, blocks["acidity"]["col"]] = m[:, blocks["acidity"]["col"]] * weights["acidity"]
    m[:, blocks["body"]["col"]]    = m[:, blocks["body"]["col"]]    * weights["body"]
    m[:, blocks["abv"]["col"]]     = m[:, blocks["abv"]["col"]]     * weights["abv"]
    return m.tocsr()


def _apply_weights_vec(vec: np.ndarray, blocks: dict, weights: dict) -> np.ndarray:
    """Block-scale a single dense vector (the seed) the same way _apply_weights
    scales the matrix, so the seed lives in the same weighted space as the rows."""
    out = vec.astype(np.float64, copy=True)
    g, r = blocks["grape"], blocks["region"]
    out[g["start"]:g["end"]] *= weights["grape"]
    out[r["start"]:r["end"]] *= weights["region"]
    out[blocks["acidity"]["col"]] *= weights["acidity"]
    out[blocks["body"]["col"]]    *= weights["body"]
    out[blocks["abv"]["col"]]     *= weights["abv"]
    return out


def cb_available() -> bool:
    return _load() is not None


def get_blocks() -> dict | None:
    """Expose the matrix layout + grape vocab so callers can build a seed vector
    that lines up with the trained matrix columns. None if CB isn't built."""
    st = _load()
    if st is None:
        return None
    return {"dim": st["dim"], "blocks": st["blocks"], "grape_vocab": st["grape_vocab"]}


def cb_scores(liked, candidate_ids, weights: dict | None = None,
              seed_vec: np.ndarray | None = None,
              seed_weight: float = 0.0) -> dict[int, float]:
    """
    Cosine similarity between the user's taste profile and each candidate.

    liked         : list of (wine_id, rating)
    candidate_ids : iterable of wine_id to score
    seed_vec      : optional cold-start seed (UNWEIGHTED, in the matrix layout)
                    inferred from the user's fruit picks. Folded into the profile
                    as one weighted-mean term so it decays as ratings accumulate:

        profile = (seed_weight*seed + sum_i (r_i-3)*row_i) / (seed_weight + sum_i |r_i-3|)

    seed_weight   : strength of the seed (~2.0 == worth two strong ratings).

    Returns {wine_id: cosine in [-1, 1]} (0.0 for wines not in the matrix or
    when there is no usable taste profile). With no seed it is byte-for-byte the
    previous rating-only behavior.
    """
    st = _load()
    if st is None:
        return {int(c): 0.0 for c in candidate_ids}

    w = weights or DEFAULT_WEIGHTS
    wmat = _apply_weights(st["mat"], st["blocks"], w)
    row_of = st["row_of"]

    # taste profile: rating-weighted mean of liked wine rows (weight = rating-3),
    # plus an optional fruit seed term in the same weighted space.
    prof = None
    wsum = 0.0
    for wine_id, rating in (liked or []):
        i = row_of.get(int(wine_id))
        if i is None:
            continue
        coeff = float(rating) - 3.0
        if coeff == 0.0:
            continue
        vec = wmat.getrow(i).toarray().ravel() * coeff
        prof = vec if prof is None else prof + vec
        wsum += abs(coeff)

    if seed_vec is not None and seed_weight > 0.0:
        sv = _apply_weights_vec(np.asarray(seed_vec), st["blocks"], w) * seed_weight
        prof = sv if prof is None else prof + sv
        wsum += seed_weight

    if prof is None or wsum == 0.0:
        return {int(c): 0.0 for c in candidate_ids}
    prof /= wsum
    pn = np.linalg.norm(prof)
    if pn == 0.0:
        return {int(c): 0.0 for c in candidate_ids}

    out: dict[int, float] = {}
    for c in candidate_ids:
        i = row_of.get(int(c))
        if i is None:
            out[int(c)] = 0.0
            continue
        v = wmat.getrow(i).toarray().ravel()
        vn = np.linalg.norm(v)
        out[int(c)] = float(prof @ v / (pn * vn)) if vn else 0.0
    return out


def pairwise_similarity(wine_ids: list[int]) -> dict[tuple[int, int], float]:
    """
    Pairwise CB cosine similarity between a list of wines.
    Returns {(min_id, max_id): cosine} for all pairs. Used by MMR.
    """
    st = _load()
    if st is None:
        return {}
    wmat = _apply_weights(st["mat"], st["blocks"], DEFAULT_WEIGHTS)
    row_of = st["row_of"]
    vecs: dict[int, np.ndarray] = {}
    for wid in wine_ids:
        i = row_of.get(int(wid))
        if i is None:
            continue
        v = wmat.getrow(i).toarray().ravel()
        n = np.linalg.norm(v)
        if n > 0:
            vecs[int(wid)] = v / n
    out: dict[tuple[int, int], float] = {}
    ids = list(vecs.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            out[(min(a, b), max(a, b))] = float(vecs[a] @ vecs[b])
    return out
