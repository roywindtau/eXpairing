# HLD — Personalized "Recommend me a wine"

## Problem

The "Suggest me a wine" button currently returns the **top-10 most popular wines**
(Bayesian-smoothed `avg_rating`/`n_ratings`), identical for every user.
`GET /wine/ranked` takes no `user_id`, so nothing about who the user is, what they
drank, or what they liked changes the result.

We want the button to return the **top-5 wines personalized** to the user, blending
two signals:

- **wine-CF** — `taste(user, wine)` from collaborative filtering (already trained:
  confidence-weighted ALS, `models/wine_als_*`, NDCG@10 = 0.029).
- **wine↔wine CB** — content similarity between a wine and the user's liked wines
  (style, grape, food-pairing, body tokens). **Training script exists
  (`backend/ml/wine/training/train_cb.py`) but has never been run and there is no
  serving code.**

The open design questions the task names:
1. **Cold start** — what does a user with 0–few ratings get?
2. **Blending** — how do CF and CB combine into one ranked list?

## Solution

Introduce a **wine scoring service** that ranks candidate wines by a calibrated blend
of CF and CB, with a **popularity-driven cold start** that ramps smoothly into the
personalized blend as the user rates more wines. This mirrors the recipe module's
proven cold→warm soft-blend pattern, so the architecture is already validated in this
codebase.

**Cold-start decision: start with popularity, ramp in personalization.** New users
(0 ratings) get today's popularity list unchanged — no regression, always works. As
ratings accumulate, CF and CB weights ramp up and popularity's weight ramps down. We do
**not** build a separate cold-start model; popularity *is* the cold-start model. This
matches the X-Wines reality (0.02% density, explicit ratings only, no implicit events),
where a brand-new user genuinely has no personal signal.

**Per-user state thresholds** (mirroring recipes' 5-rating warm threshold):

| User state | Ratings | What ranks the list |
|---|---|---|
| Cold | 0 | 100% popularity (current behavior) |
| Warming | 1–4 | CB (from liked wines) + popularity; CF ramping in |
| Warm | ≥ 5 | CF + CB blend, popularity as tiebreak/floor |

**Blend formula** (calibrated, min-max normalized per candidate pool — same calibration
discipline as recipes):

```
score(user, wine) = w_cf · cf(user, wine)
                  + w_cb · cb(user, wine)
                  + w_pop · popularity(wine)
```

Weights are a function of `n_ratings(user)`: `w_pop` decays, `w_cf`/`w_cb` grow. CB is
available earlier than CF (a single liked wine gives a CB profile; CF needs the user in
the factor matrix or a fold-in). Exact weight schedule is tuned offline, not guessed —
seeded from recipe defaults (CF highest) and validated on the frozen wine split.

### Flow

```
click "Suggest me a wine"
        │
        ▼
GET /wine/ranked?user_id=U&top_n=5
        │
        ▼
load user's wine ratings (WineEvent, synthetic=False)
        │
        ├─ 0 ratings ──────────────► popularity top-N        (cold)
        │
        └─ ≥1 ratings:
              build candidate pool (popular + CB neighbors of liked wines)
                       │
              ┌────────┴─────────┐
              ▼                  ▼
        cf(user,wine)      cb(user,wine)
        (ALS factors)      (TF-IDF cosine vs liked-wine profile)
              └────────┬─────────┘
                       ▼
              calibrate (min-max) + weighted blend by n_ratings
                       ▼
                 top-5 wines  ──►  WineCard feed
```

## Sketch

Pseudocode only — shape of the change, not implementation.

```
# backend/ml/wine/serving/serve_cb.py   (NEW)
load wine_cb_matrix, wine_cb_ids, vectorizer at import
function cb_scores(liked_wine_ids, liked_ratings, candidate_ids):
    if no liked_wines: return zeros
    profile = weighted_mean( cb_vector(w) for w in liked_wines,
                             weight = rating - 3.0 )      # taste profile, like recipes
    for each candidate: score = cosine(profile, cb_vector(candidate))
    return scores

# backend/ml/wine/serving/serve_cf.py   (NEW)
load wine_als_model (user_factors, item_factors, id maps) at import
function cf_scores(user_id, candidate_ids):
    if user_id in user_factor_map:
        u = user_factors[user_id]
    else:
        u = fold_in(liked_wine_ids)        # cold-fold for users not in training
    for each candidate: score = dot(u, item_factors[candidate])
    return scores

# backend/services/wine/scoring.py   (NEW)
function rank_wines(user_id, top_n):
    liked = wine_events(user_id, synthetic=False)
    if len(liked) == 0:
        return popularity_top_n(top_n)            # cold start
    candidates = popularity_pool() ∪ cb_neighbors(liked)
    cf  = cf_scores(user_id, candidates)
    cb  = cb_scores(liked, candidates)
    pop = popularity(candidates)
    w_cf, w_cb, w_pop = weight_schedule(len(liked))    # ramp by rating count
    blended = w_cf·norm(cf) + w_cb·norm(cb) + w_pop·norm(pop)
    return top_n(candidates by blended)

# backend/routers/wine.py   (CHANGED)
GET /wine/ranked?user_id=&top_n=
    if user_id is None: popularity_top_n(top_n)   # back-compat
    else:               rank_wines(user_id, top_n)
```

## Files added and changed

**Added**
- `backend/ml/wine/training/train_cb.py` — *exists; needs to be run* to produce the CB
  artifacts (`wine_cb_matrix.npz`, `wine_cb_ids.npy`, `wine_cb_vectorizer.pkl`).
- `backend/ml/wine/serving/serve_cb.py` — load CB artifacts; build a liked-wine taste
  profile; cosine-score candidates.
- `backend/ml/wine/serving/serve_cf.py` — load ALS factors; score candidates per user;
  fold-in for users absent from training.
- `backend/services/wine/scoring.py` — candidate pool, calibration, weight schedule,
  the cold→warm blend.
- `tests/wine/test_wine_scoring.py` — cold/warming/warm routing, weight ramp,
  calibration, fold-in.

**Changed**
- `backend/routers/wine.py` — `/wine/ranked` accepts optional `user_id`; routes to the
  scoring service when present, popularity when absent.
- `frontend/src/api/wine.ts` + `WineForYouPage.tsx` — pass `user_id`; surface the
  CF/CB/popularity strategy (mirrors the recipe feed's CF-strategy banner).
- `README.md` — update the Wine recommender section once CB+CF are wired.

## New dependencies

**None.** ALS artifacts already exist; CB uses `scikit-learn` TF-IDF (already a
dependency for recipe CB); the `implicit` library used to train ALS is only needed
offline. Serving is pure NumPy/SciPy dot products and cosine — already in `requirements.txt`.

## Alternative solutions

**A. Hard switch at a rating threshold (popularity below N, blend above).**
- *Pros:* simplest; no weight schedule to tune.
- *Cons:* visible "jump" in recommendations the moment the user crosses the threshold;
  wastes the 1–4 ratings that already carry CB signal. The recipe module deliberately
  moved away from this to a soft blend.

**B. CF-only personalization (skip CB for v1).**
- *Pros:* fastest to ship — CF is already trained; no CB run needed.
- *Cons:* misses the task's stated end goal (two models); CF alone can't explain *why*
  a wine fits ("because you liked Malbecs"); ALS cold-fold is weak for very new users
  where CB from one liked wine is actually stronger.

**C. Build a dedicated cold-start model (e.g. content seeds from a taste questionnaire).**
- *Pros:* personalization from rating #0.
- *Cons:* no onboarding taste questionnaire exists for wine; building one is a separate
  feature. Popularity is a well-understood, zero-cost cold start that the data density
  justifies.

## Why this solution

- **Reuses a validated pattern.** The recipe module already ships a calibrated, soft
  cold→warm CF+CB blend with MMR and min-max calibration. Mirroring it for wine means
  the architecture is de-risked and the team already understands it.
- **No regression, ever.** Cold users get exactly today's popularity list; personalization
  is purely additive as signal arrives. Back-compat is preserved by making `user_id`
  optional.
- **Matches the data.** X-Wines is explicit-only and brutally sparse, so popularity is
  the honest cold-start prior and ALS (already chosen for its ranking strength over SVD)
  is the right warm engine. CB fills the gap ALS is weakest in — the 1–4 rating window.
- **Cheap to serve.** No new infra or dependencies; all serving is dot-products and
  cosine over artifacts that exist or are one training run away.

## Open items to resolve before/while building

1. **Run `train_cb.py`** and sanity-check the CB artifacts (vocab size, that
   `harmonize_csv`/`grapes_csv` are populated on seeded wines).
2. **Confirm ALS id-map shape** in `wine_als_model.npz` to decide fold-in vs. direct lookup.
3. **Tune the weight schedule** on `models/wine_split/` (don't guess CF/CB/pop weights).
4. **Decide candidate-pool size** (e.g. top-200 popular ∪ CB neighbors) to keep
   calibration meaningful and serving fast.
