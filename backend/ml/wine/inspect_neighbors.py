"""
inspect_neighbors.py
--------------------
Throwaway diagnostic — NOT the production serving path. Builds the structured
weighted CB vector for every wine and prints nearest neighbors for a handful of
seed wines, with a per-block breakdown of WHY each neighbor matched.

The point: eyeball whether the sommelier's palate-first weights produce
wine-sensible neighbors before we commit to serve_cb.py.

Spec (see memory wine-cb-design):
  weights (sommelier, normalized): acidity .368, body .368, region .158,
                                   abv .053, grape .053
  encoding: acidity/body ordinal, region one-hot on rolled-up parent,
            abv normalized+clipped, grape multi-hot. Each block unit-normalized
            then × weight. style = HARD FILTER (neighbors share the seed's style).

Run:
    python -m backend.ml.wine.inspect_neighbors
    python -m backend.ml.wine.inspect_neighbors --seeds 100200 100300
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from backend.db.database import SessionLocal
from backend.db.models import Wine

WEIGHTS = {"acidity": 0.368, "body": 0.368, "region": 0.158, "abv": 0.053, "grape": 0.053}

ACIDITY = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
BODY = {"Very light-bodied": 0.0, "Light-bodied": 0.25, "Medium-bodied": 0.5,
        "Full-bodied": 0.75, "Very full-bodied": 1.0}
ABV_LO, ABV_HI = 5.0, 16.0          # clip band for the dirty 0..50 raw range

ROLLUP = json.loads((Path("models") / "region_rollup.json").read_text(encoding="utf-8"))


def _grapes(w: Wine) -> list[str]:
    return [g.strip() for g in (w.grapes_csv or "").split(",") if g.strip()]


def build_vectors(wines: list[Wine]):
    """Return (matrix, meta) — matrix is n_wines x dim, already weighted."""
    # vocabularies
    grapes = sorted({g for w in wines for g in _grapes(w)})
    regions = sorted({ROLLUP.get(w.region, w.country or "?") for w in wines if w.region})
    g_idx = {g: i for i, g in enumerate(grapes)}
    r_idx = {r: i for i, r in enumerate(regions)}

    rows = []
    for w in wines:
        # --- grape block (multi-hot, unit-normalized) ---
        gv = np.zeros(len(grapes))
        for g in _grapes(w):
            gv[g_idx[g]] = 1.0
        if gv.any():
            gv /= np.linalg.norm(gv)
        gv *= WEIGHTS["grape"]

        # --- region block (one-hot, already unit-length) ---
        rv = np.zeros(len(regions))
        if w.region:
            rv[r_idx[ROLLUP.get(w.region, w.country or "?")]] = 1.0
        rv *= WEIGHTS["region"]

        # --- ordinal/numeric scalars (each already its own normalized number) ---
        acid = ACIDITY.get(w.acidity, 0.5) * WEIGHTS["acidity"]
        body = BODY.get(w.body, 0.5) * WEIGHTS["body"]
        abv_raw = min(max(w.abv if w.abv is not None else 13.0, ABV_LO), ABV_HI)
        abv = (abv_raw - ABV_LO) / (ABV_HI - ABV_LO) * WEIGHTS["abv"]

        rows.append(np.concatenate([gv, rv, [acid, body, abv]]))

    mat = np.vstack(rows)
    meta = {"grapes": grapes, "regions": regions,
            "g_dim": len(grapes), "r_dim": len(regions)}
    return mat, meta


def block_breakdown(va, vb, meta):
    """Per-block dot-product contributions to the (unnormalized) similarity."""
    gd, rd = meta["g_dim"], meta["r_dim"]
    g = float(va[:gd] @ vb[:gd])
    r = float(va[gd:gd + rd] @ vb[gd:gd + rd])
    a = float(va[gd + rd] * vb[gd + rd])
    b = float(va[gd + rd + 1] * vb[gd + rd + 1])
    al = float(va[gd + rd + 2] * vb[gd + rd + 2])
    return {"grape": g, "region": r, "acidity": a, "body": b, "abv": al}


def fmt(w: Wine) -> str:
    return (f"{w.name[:34]:34} | {w.style:9} | {w.body or '?':16} | "
            f"acid={w.acidity or '?':6} | {(w.grapes_csv or '')[:24]:24} | "
            f"{ROLLUP.get(w.region, w.country or '?')}")


def main(seed_ids: list[int]):
    db = SessionLocal()
    try:
        wines = db.query(Wine).all()
    finally:
        db.close()
    by_id = {w.id: w for w in wines}
    idx_of = {w.id: i for i, w in enumerate(wines)}

    mat, meta = build_vectors(wines)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0

    if not seed_ids:
        # pick a few diverse seeds: a full-bodied red, a crisp white, a sparkling
        def pick(pred):
            for w in wines:
                if pred(w):
                    return w.id
        seed_ids = [s for s in [
            pick(lambda w: w.style == "Red" and w.body == "Full-bodied" and w.acidity == "High"),
            pick(lambda w: w.style == "White" and w.acidity == "High" and w.body == "Light-bodied"),
            pick(lambda w: w.style == "Sparkling"),
        ] if s]

    for sid in seed_ids:
        seed = by_id.get(sid)
        if seed is None:
            print(f"\n!! seed {sid} not found"); continue
        i = idx_of[sid]
        sims = (mat @ mat[i]) / (norms * norms[i])

        # STYLE HARD FILTER: only same-style candidates
        same_style = np.array([w.style == seed.style for w in wines])
        sims = np.where(same_style, sims, -1.0)
        sims[i] = -1.0  # exclude self

        top = np.argsort(sims)[::-1][:5]
        print("\n" + "=" * 110)
        print("SEED:  " + fmt(seed))
        print(f"       (style-filtered to '{seed.style}' candidates)")
        print("-" * 110)
        for j in top:
            w = wines[j]
            bd = block_breakdown(mat[i], mat[j], meta)
            tot = sum(bd.values()) or 1e-9
            share = {k: f"{100*v/tot:4.0f}%" for k, v in bd.items()}
            print(f"  sim={sims[j]:.3f}  " + fmt(w))
            print(f"     why: grape {share['grape']}  region {share['region']}  "
                  f"acidity {share['acidity']}  body {share['body']}  abv {share['abv']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    seeds = []
    if "--seeds" in args:
        seeds = [int(x) for x in args[args.index("--seeds") + 1:]]
    main(seeds)
