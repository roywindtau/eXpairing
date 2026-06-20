# Wine CB — `wine-cb` branch summary

What the `wine-cb` branch built: the **content-based (CB) half** of the wine
recommender, wired into a personalized **"Suggest me a wine"** feature that
blends CB with the already-trained collaborative-filtering (ALS) model and a
popularity cold start.

Companion docs: [wine-cf-experiments.md](wine-cf-experiments.md) (the CF/ALS
model decision). This file covers the CB model and the end-to-end personalization.

---

## Goal

Turn the naive button (top-10 by popularity, no personalization) into a
personalized feed driven by two models — **CF** (taste from co-rating) and
**CB** (taste from wine attributes) — with sensible cold-start behavior.

---

## Key design decisions

### CB is a structured weighted vector — not TF-IDF, not embeddings
X-Wines has **zero free text** (verified against the raw CSVs and the ratings
file — no tasting notes, no reviews). Every wine attribute is a structured
categorical/ordinal tag. So:

- **Embeddings** (sentence-transformers / OpenAI) were considered and rejected:
  they shine on free-text synonymy (Chenin vs. Sauvignon, Sonoma ≈ Loire), but
  the data is controlled tags where that nuance barely exists. Overkill, opaque,
  and an extra dependency.
- **TF-IDF** was rejected too — lexical matching on tags adds nothing over
  direct one-hot, and it can't represent ordinals or numerics correctly.
- **Chosen:** a structured feature vector, one block per attribute, weighted and
  compared by cosine. Pure NumPy/SciPy, sparse, interpretable, no new deps.

### Encoding (per attribute)
| Field | Encoding |
|---|---|
| grape | multi-hot (777 grapes), block unit-normalized |
| region | one-hot on a **rolled-up parent** region (see below) |
| acidity | ordinal Low/Med/High → 0 / 0.5 / 1 |
| body | ordinal 5 levels → 0…1 |
| abv | min-max normalized, clipped to a sane [5,16] band (raw data is dirty) |
| style | **not in the vector** — a hard pre-filter |
| country | folded into region as its fallback level |

### Weights = sommelier's palate-first prior (not learned)
CB has **no learning** — the per-block weights are fixed constants. They came
from a real sommelier, framed as "split 100% by what predicts you'll like
another wine," normalized:

```
acidity 36.84   body 36.84   region 15.79   abv 5.26   grape 5.26
```

Grape is deliberately **low**: the sommelier's view is that *structure*
(acidity + body, ~74% of the weight) drives palate similarity more than variety.
Each block is unit-normalized, then scaled by its weight; weights are applied at
**serve time** (not baked into the saved matrix) so they can be retuned — or
overridden per request — without recomputing 100k vectors.

### Style is a hard filter; CF + popularity break CB's ties
CB intentionally clusters wines into coarse structural buckets (only 3 acidity ×
5 body combos carry most of the weight), so many wines tie at cosine ≈ 1.0.
That's **by design** — CB picks the structural shortlist; **CF and popularity
rank within it**. CB is a blend term, not a filter. The only hard gate is
**style** (a red-drinker never sees whites unless they ask).

### Cold start = popularity (no separate model)
- **0 ratings** → top popularity (no regression, always works).
- **1–4 ratings (warming)** → CB + popularity.
- **≥5 ratings (warm)** → `0.5·CF + 0.5·CB` (min-max calibrated), popularity as
  cold-start floor. CF for a brand-new user is noise, so it's kept out until 5
  ratings.

---

## Region rollup

Raw region one-hot is useless: **2,160** appellations, the top 50 cover only
~42% of wines, so two wines almost never share an exact region.
`data/wine/region_rollup.py` collapses them to **107 parent regions**
(Pauillac → Bordeaux, Meursault → Burgundy, Napa → California…), with **country
as the fallback** for the obscure long tail. Result: 57.8% of wines land on a
real sub-region parent, 42.2% on a country fallback. Output committed-by-script
to `models/region_rollup.json`.

---

## CF fold-in (validated)

App users aren't in the ALS training factors (those are X-Wines ids), so
`serve_cf.py` **folds them in**: solve the standard ALS user update from the
wines they rated (`C = 1 + 5·rating`, reg 0.05). **Validated** via leave-one-out
on 200 real users — held-out wines rank at the **0.92 mean percentile** (median
0.978, 70% in top-5%). Not just "runs" — correct.

---

## Frontend ("Suggest me a wine")

- Button passes `user_id` → personalized ranking.
- **Style checkboxes** — user can override which styles to generate (even styles
  they've never rated).
- **Cards grouped style → pairing**: one section per wine style, sub-grouped by
  primary food pairing (most common across the row).
- **Per-style color tints** on cards and the picker chips (single
  `STYLE_COLORS` source of truth).
- Wine-bottle (🍷) loading spinner; equal-height cards; "why this wine" / "pairs
  with" clutter moved out of the cards into group labels.

---

## Files

**Backend**
- `backend/ml/wine/training/train_cb.py` — precompute the structured matrix
  (`models/wine_cb_matrix.npz` + ids + block layout), unweighted at rest.
- `backend/ml/wine/serving/serve_cb.py` — taste profile → CB cosine.
- `backend/ml/wine/serving/serve_cf.py` — ALS fold-in → CF scores.
- `backend/services/wine/scoring.py` — style filter + cold/warming/warm blend.
- `backend/routers/wine.py` — `/wine/ranked` takes `user_id` and `styles`; new
  `WineOut` fields (acidity, body, region).

**Data processing (offline, not served)**
- `data/wine/region_rollup.py` — appellation → parent map.
- `data/wine/inspect_neighbors.py` — diagnostic used to validate the weights.

**Frontend**
- `WineForYouPage.tsx`, `WineCard.tsx`, `api/wine.ts`, `index.css`.

**Tooling / tests**
- `run_backend.sh` — local uvicorn with `--reload`; `docker-compose.yml` dev
  reload + source mount.
- `tests/wine/test_wine_scoring.py` — cold/warming/warm routing, style filter,
  no-re-recommend, blend (7 tests).

---

## Artifacts (git-ignored, regenerated by scripts)
```
models/region_rollup.json     2,160 → 107 region map
models/wine_cb_matrix.npz     100,646 × 887 structured vectors (unweighted)
models/wine_cb_ids.npy        wine_id per row
models/wine_cb_blocks.json    block layout + grape/region vocab
models/wine_als_*             CF/ALS model (from the wine-cf work)
```

---

## Known limitations / future work
- **Grape labels are noisy** in X-Wines (e.g. a Cabernet blend tagged "Pinot
  Noir") — affects the grape block and any grape-based UI text.
- The pairing sub-groups are only as granular as `harmonize_csv` (coarse food
  categories).
- Weights are a sommelier prior, validated by eyeballing neighbors — not tuned
  against held-out ranking metrics (kept deliberately CF-independent).
- `top_n` is 10 in the UI; the original goal mentioned 5 — a product decision.
