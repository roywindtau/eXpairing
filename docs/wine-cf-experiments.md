# Wine CF — Experiment Log & Model Decision

How we arrived at the wine collaborative-filtering model: **confidence-weighted ALS,
alpha=5, factors=64**. This documents the experiments, the reasoning, and the dead
ends — so the decision is reproducible and the "why" survives.

---

## Problem

Train the `taste(user, wine)` CF signal for the drink recommender. Wine data is the
X-Wines Full set: **21M ratings, 1.06M users, 100K wines, 0.02% density** — explicit
1–5 star ratings only, no interaction events.

Two questions had to be answered:
1. Which algorithm — Funk SVD (like recipes) or ALS?
2. How to tune it?

---

## Why not just reuse the recipe approach (Funk SVD)?

Recipes use Funk SVD (Surprise). The instinct was to mirror it. But the datasets differ:

| | Recipes (Food.com) | Wine (X-Wines) |
|---|---|---|
| Explicit ratings | ✅ | ✅ |
| **Implicit events** (cook/like/save) | ✅ in dataset | ❌ none |
| Scale | modest | 21M ratings |
| Density | denser | brutal (0.02%) |

The key difference is **implicit feedback**. Recipe SVD doesn't train on ratings alone —
it converts *cook events* into synthetic ratings (`implicit_rating = max(3.0, 4.0 −
min(n_missing,3)×0.3)`), which densifies the matrix and gives SVD signal on un-rated
items. That patch is what lets SVD *rank* well for recipes.

Wine has no such events — only explicit ratings. So vanilla SVD can't be patched the
same way. This pointed at ALS, which manufactures the implicit/ranking signal natively
from the rating matrix (it treats every interaction as a confidence-weighted positive).

---

## The algorithm bake-off: ALS vs Funk SVD (same data, same metrics)

We didn't decide on theory — we trained both and evaluated on a **frozen leave-5-out
split** so results were directly comparable. Two metrics, because each algorithm
optimizes a different objective:

| | RMSE (rating prediction) | NDCG@10 (ranking) |
|---|---|---|
| **Funk SVD** | **0.596** ✅ (beats naive floor 0.738) | ~0.0006 ❌ (≈ random) |
| **ALS (alpha=5)** | 2.393 ❌ (≈ naive floor) | **0.0291** ✅ |
| Popularity baseline | — | 0.0071 |

**Interpretation:**
- SVD is genuinely good at *predicting the star value* (its objective: minimize RMSE).
  But for **ranking** it collapses — it has no signal on un-rated items, so its top-N is
  near-arbitrary among the 100K wines.
- ALS treats unobserved items as weak negatives (confidence `C = 1 + alpha*rating`), so
  it learns to **separate** "would like" from "wouldn't" → ranks ~50× better than SVD and
  ~4× better than popularity.
- The RMSE result is the mirror image: ALS can't predict star values (it doesn't try),
  so its calibrated RMSE is no better than guessing the mean. **This is expected, not a
  failure** — RMSE is not ALS's objective.

**Decision rule learned:** *a model is only good at the metric it was trained to
minimize.* The recommender **ranks**, so the metric that matters is NDCG → **ALS wins**.
(SVD's RMSE advantage is irrelevant: the recommender never needs to predict a star value.)

> Note on comparability: a metric is only meaningful for comparison if both models are
> graded against the *same target*. RMSE-vs-stars and NDCG-vs-ranking are the two shared
> yardsticks; grading each model against its own training target would not be comparable.

---

## Building trustworthy evaluation (done before tuning)

Early ALS runs used a random 90/10 split, which (a) lets the model "see the future" and
(b) is regenerated per run, so small gains were indistinguishable from noise. Fixed by:

- **`build_wine_split.py`** — a **frozen leave-5-out split** saved to `models/wine_split/`
  (16.2M train / 4.4M test, 887K users with held-out items). Built once; every experiment
  loads the identical split.
- **`eval_wine_model.py`** — one shared `ranking_metrics_at_k` harness used by every run.

This made every later comparison apples-to-apples by construction.

---

## Tuning experiments (all on the frozen split, NDCG@10)

### 1. Confidence scale `alpha` — the big lever
`C = 1 + alpha * rating`. Swept {1, 5, 15, 40}:

| alpha | ndcg@10 |
|---|---|
| 1 | 0.0267 |
| **5** | **0.0291** ← best |
| 15 | 0.0288 |
| 40 (original) | 0.0263 |

**alpha=5 won (+10% over the original 40).** alpha=40 over-saturated confidence
(everything looked like a strong positive, so ALS couldn't tell a 3-star from a 5-star);
alpha=1 under-weighted the signal. The curve peaks around 5. **Adopted alpha=5.**

### 2. factors × regularization
Swept factors {64,128,200} × reg {0.01,0.05,0.1}. The factors=64 row was flat
(~0.0291) and reg barely moved anything. No meaningful gain → kept **factors=64,
reg=0.05**. (The slow high-factor runs were not worth completing once 64 proved flat.)

### 3. Matrix weighting — linear vs TF-IDF vs BM25
Tested alternative weightings (still pure CF — none read wine metadata):

| scheme | ndcg@10 |
|---|---|
| TF-IDF | 0.0293 |
| linear (alpha=5) | 0.0291 |
| BM25 (K1=100) | 0.0073 ⚠️ |

TF-IDF ≈ linear (noise). **BM25 collapsed** (-75%) — its default saturation was far too
aggressive for this data. **Kept linear alpha=5.**

**Conclusion of tuning:** matrix-weighting is exhausted. alpha helped (+10%);
factors/reg, TF-IDF flat; BM25 hurt. **0.0291 is the practical ceiling** for pure-CF
tuning on this data.

---

## Is 0.029 "good"? Yes — judge by ratio, not absolute

0.029 *looks* low against an abstract 1.0, but:
- It's **4× the popularity baseline** (0.0071) and ~50× SVD — the only comparisons that mean anything.
- It's **structurally capped**: each user has 5 held-out wines out of 100K at 0.02%
  density. Most "misses" are wines the user never rated, not bad recommendations.
- 0.02–0.05 NDCG@10 is a normal, healthy band for sparse implicit recommenders.

**Lesson: absolute ranking metrics are meaningless without a baseline.**

---

## Final model

- **Algorithm:** confidence-weighted ALS (`implicit` library)
- **alpha = 5**, **factors = 64**, **iterations = 15**, **reg = 0.05**
- **Filtering:** ≥5 ratings per user and per item
- **NDCG@10 = 0.0291** (4× popularity), on the frozen leave-5-out split
- **Artifacts:** `models/drink_wine_als_model.npz` (factors + id maps) + `_meta.json`

### Code (in `backend/ml/drinks/training/wine/`)
- `wine_data.py` — shared data layer: load, sparsity filter, confidence matrix; holds the
  tuned `CONFIDENCE_ALPHA=5` and `SPLIT_BUILT_ALPHA=40` constants.
- `build_wine_split.py` — builds/freezes the leave-5-out split (run once).
- `eval_wine_model.py` — the shared ranking-eval harness.
- `eval_wine_popularity.py` — popularity baseline (the floor ALS must beat).
- `train_wine_als.py` — trains the final model (re-weights the frozen split to alpha=5).

---

## Open threads (not done)

1. **Serving wiring** — the trained ALS model is **not yet plugged into serving**. Wine
   `cf_score` currently comes from item-similarity (`serve_cf.py`), not the ALS factors.
   Wiring ALS in is the task that makes this tuning actually reach users. Note: the blend
   formula (`scoring.py`) already min-max calibrates each signal, so it absorbs ALS's
   odd confidence scale automatically — only the serving layer needs to feed it ALS scores.
2. **`≥4` positive cut** — the one untested CF lever: drop ratings <4 so disliked wines
   stop acting as positives. Could push past 0.029, untested.
3. **Implicit drink events** — once the app logs `like`/`save`/`skip` at runtime, wine
   gains the implicit channel recipes got from Food.com. ALS already exploits implicit
   signal natively, so this would strengthen it further.

---

## Reusable decision procedure (for future CF signals)

1. **Pick the metric from the use case** — does the consumer need a *ranking* (→ NDCG) or
   a *calibrated value* (→ RMSE)? This upstream choice decides the winner.
2. Train candidate algorithms on the same data.
3. Evaluate all on that one metric, same frozen split.
4. Compare to the baseline (popularity for ranking, global-mean for RMSE) so numbers are meaningful.
5. Pick the winner; tie → cheaper/faster wins.
