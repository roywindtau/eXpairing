# Project Summary

This document records how eXpairing was actually built: what we tried, what failed, what we concluded from each failure, and why the shipped version looks the way it does. Where the [Modules Description](modules.md) document describes the *final* system, this one describes the path to it — including the branches we abandoned.

## Development Evolution

Development proceeded in five milestones. Each one was driven less by a feature checklist than by a specific problem that the previous milestone exposed.

### Milestone 1 — Foundations & Rule-Based MVP
We began with the simplest thing that could produce a ranked feed: a SQLite schema, FastAPI scaffolding, a React single-page interface, and a candidate fetcher that ranked recipes purely by rule-based ingredient overlap — count how many of the recipe's ingredients you have, sort descending.

**What we learned**: pure overlap ranking is degenerate. It surfaces recipes with two or three ingredients (which you trivially "have"), and it has no notion of whether the user would actually enjoy the result. Overlap is a *feasibility* signal, and feasibility alone is not a recommendation. This framed the rest of the project as a problem of combining feasibility with preference, rather than optimizing either in isolation.

### Milestone 2 — Content-Based Engine & Domain Constraints
We fitted a `TfidfVectorizer` over the 231k Food.com recipes, implemented exponential expiry urgency decay ($\exp(-k \cdot \text{days})$) over pantry dates, and introduced the per-user $\beta$ parameter to penalize recipes with missing ingredients.

**The decision that mattered here was TF-IDF over dense embeddings**, and it went against the obvious instinct. Pre-trained embeddings (Word2Vec, BERT, sentence transformers) are the default choice for text similarity precisely because they capture semantic proximity: `butter ≈ margarine`, `scallion ≈ spring onion`. We prototyped with that assumption and found it is *actively wrong* for an inventory-matching system. If a user has butter in the fridge and butter is about to expire, the system must surface recipes that call for **butter**. A recipe calling for margarine is not a near-miss — it is a recipe the user cannot cook, and boosting it directly undermines the product's entire premise. Semantic substitution is a feature in search and a bug in inventory matching. TF-IDF over a canonical ingredient vocabulary gives exact lexical matching, and the IDF term supplies the discrimination we actually wanted for free: it down-weights salt and flour while up-weighting saffron and miso. We kept it, and it still captures cuisine affinity (miso + soy sauce clustering toward Japanese dishes) without any cuisine tag metadata.

### Milestone 3 — Collaborative Filtering & the Cold-Start Problem
We trained Biased Funk SVD on the 1.1M explicit Food.com ratings via scikit-surprise (50 latent factors, SGD), reaching **RMSE 0.6136** against a global-mean baseline of 1.12 — a 45% error reduction.

This immediately created a new problem: matrix factorization is useless for a user with no ratings, and *every* new eXpairing user has no ratings. The obvious fallback — serve a popularity list — was unacceptable, because a generic top-recipes list is exactly the non-personalized experience the product exists to replace.

The approach we settled on was **preference-seeded item similarity**. We built a sparse item-item cosine similarity matrix from mean-centered co-rating patterns across 51k+ recipes, then used the cold user's *pantry contents and dietary tags* as preference anchors into that matrix. The insight is that a cold user is not truly information-free: their fridge is a preference signal. This lets a content-side signal bootstrap the new user directly into the behavioral item graph.

We then had to handle the boundary. A hard switch at 5 ratings produced a visible discontinuity — a user's feed would lurch on their fifth rating. We replaced it with a linear blend, `alpha = n_ratings / 5`, over the 1–4 rating band, and verified via lifecycle simulation (`evaluate.py --lifecycle`, stepping a synthetic user from 0 to 10 ratings) that scores ramp smoothly with no jump at the handover.

### Milestone 4 — Score Calibration
This milestone exists because of a bug that did not look like a bug.

We had four signals (CF, CB, expiry urgency, ingredient match) and a weighted sum with weights we had reasoned about carefully. The feed came back dominated almost entirely by expiry urgency, regardless of what we set the weights to. Lowering the expiry weight barely changed the ordering.

The cause was **distribution scale, not weights**. Raw CF scores across a candidate pool cluster in a very narrow band — roughly 0.30 to 0.38. Expiry urgency, by construction, spans nearly the full unit interval, 0.02 to 0.95. When you sum them, the CF term contributes at most ~0.08 of spread while expiry contributes ~0.93. The nominal weight of 35% on CF was multiplying a variable that had almost no variance. The weights were correct on paper and meaningless in practice.

The fix is **per-candidate min-max normalization**: each score component is normalized across the current candidate pool *before* blending, so every component contributes a full $[0, 1]$ range and the assigned weights (γ=0.35 CF, α=0.35 expiry, β=0.20 match, δ=0.10 CB) hold their intended proportions. A grid search over γ and α confirmed those defaults after calibration was in place — a search that would have been worthless before it, since the weights were not the operative variable.

The generalizable conclusion, and the single most useful thing we took from this project: **in a hybrid recommender, you cannot sum scores from different models without calibrating their distributions first.** Blending is a statement about relative influence, and relative influence is determined by variance, not by the coefficient you wrote down.

We also added MMR reranking (λ=0.7, ingredient Jaccard similarity, over the top 60 candidates) in this milestone, after observing that a well-calibrated feed could still return twenty near-identical muffin variants. Relevance without diversity is monotony.

### Milestone 5 — Implicit Feedback, the Wine Domain & Pairing
The final milestone closed the feedback loop and added the second domain.

**Implicit feedback**: users cook far more often than they rate. We convert cook events into synthetic ratings via `max(3.0, 4.0 - n_missing*0.3)` — cooking something is mild positive evidence, and cooking it *despite* missing ingredients is stronger evidence still. We also built `beta_updater.py`, a daily EMA batch job (`new_β = 0.85·current_β + 0.15·revealed_β`) that drifts the user's stated waste-aversion toward their revealed behavior. This was motivated by an observed **aspirational bias**: users set a $\beta$ claiming they will cook whatever reduces waste, then in practice consistently skip recipes with missing ingredients. Rather than silently overriding the stated value, the profile surfaces a warning when stated and revealed $\beta$ diverge by $>10\%$ — the system adapts, and tells the user it is adapting.

**The wine domain** brought its own algorithm selection problem, described in full below. We integrated X-Wines, built the region rollup, ran the ALS-vs-SVD bake-off, deployed personalized wine recommendations with runtime fold-in, and engineered the 12-dimensional recipe-wine pairing engine (`serve_pairing.py`). Multi-modal vision scanning with dual provider support (`OPENAI_API_KEY` / `GEMINI_API_KEY`) plus a deterministic offline mock shipped alongside.

&nbsp;<br>

## The Wine CF Bake-Off: ALS vs Funk SVD

The wine module is where we made our most consequential — and least intuitive — modeling decision, and it is worth recording in detail because the naive reading of the numbers points the wrong way.

### The setup
We froze a leave-5-out split (`models/wine_split/`, 16.2M train / 4.4M test ratings) so every subsequent experiment was measured on identical data. We had already shipped Biased Funk SVD successfully in the recipe domain, so the working assumption was that we would reuse it. X-Wines has 21M explicit 1–5 star ratings; on paper it is exactly the matrix Funk SVD was designed for.

### What happened
Funk SVD did what it is built to do. It achieved a **rating RMSE of 0.596** — better, in absolute terms, than our recipe model's 0.6136. By the metric that had guided the entire recipe module, it was an unambiguous success.

Then we measured ranking quality. **NDCG@10 came in at approximately 0.0006.** The popularity-only baseline scores 0.0071. Funk SVD was not merely worse than popularity — at roughly a tenth of the popularity floor, it was **indistinguishable from random ordering**.

Confidence-weighted ALS ($C = 1 + \alpha \cdot \text{rating}$, factors=64, reg=0.05), by contrast, reached **NDCG@10 of 0.0291** — 4× the popularity baseline.

### Why the model with better RMSE was the worse recommender
RMSE asks: *given that this user rated this wine, how close was your predicted star value?* It is measured exclusively over observed pairs. Funk SVD trains only on observed ratings and is therefore optimizing precisely that quantity — and it optimized it well.

But the serving task is different. Serving asks: *out of 100,646 wines this user has never rated, which ten should we show?* Funk SVD has no training signal about un-rated items at all. Its predictions over the unobserved mass are unconstrained extrapolation, and there is no reason for that extrapolation to be well-ordered. A model can be excellent at interpolating within the observed support and worthless at ranking outside it.

ALS with confidence weighting treats every unobserved cell as a weak negative with low confidence, and observed ratings as positives with confidence scaling in the rating value. This means the un-rated mass is *in the objective*. The model is explicitly trained to push observed items above un-rated ones — which is the ranking problem, stated directly.

**The conclusion we drew, and the one we would carry into any future recommender: match the training objective to the serving task, not to the shape of the data.** Explicit star ratings tempt you toward a rating-prediction objective. If what you ship is a ranked list, you must optimize ranking, and you must evaluate with a ranking metric — because a rating-accuracy metric will confidently tell you that a random-ordering model is your best one.

### Hyperparameter experiments (all on the same frozen split)
Having chosen ALS, we swept its parameters:

- **Confidence scale α over {1, 5, 15, 40}** — **α=5 won**, at NDCG@10 0.0291. This is +10% over α=40, the `implicit` library default. The default over-saturates: at α=40 a 5-star rating carries confidence 201 versus an unobserved cell's 1, and the model effectively stops distinguishing between a good wine and a great one. α=1 under-weights the positive signal. α=5 sits at the balance point for this data's rating distribution.
- **factors {64, 128, 200} × regularization {0.01, 0.05, 0.1}** — essentially **flat**, with factors=64 as good as anything larger. Additional capacity bought nothing, which tells us the ceiling here is set by the signal in the data, not by model expressiveness.
- **Alternative matrix weightings** — TF-IDF weighting of the confidence matrix landed within noise of linear weighting. **BM25 weighting collapsed, −75%.** BM25's default saturation parameters are tuned for term-frequency distributions in document retrieval and are far too aggressive for a ratings matrix; they flatten the confidence signal we had just spent a sweep calibrating.

Together these established **linear confidence weighting at α=5 as the practical ceiling for pure CF on this dataset**, and told us that further gains would have to come from somewhere other than tuning.

### Fold-in validation for real app users
One gap remained. The offline ALS factors are computed over X-Wines users; an eXpairing app user does not exist in that factorization. At serving time we fold them in — solving the ALS user-factor update against the frozen item factors, using only their in-app ratings.

Fold-in is easy to get subtly wrong in a way that produces plausible-looking output: a poorly-conditioned solve tends to return something close to the global mean, which surfaces popular wines, which *look* like reasonable recommendations. We needed to prove the fold-in was producing genuine personalization rather than a popularity echo.

We validated with leave-one-out over 200 real app users: hold out one of the user's rated wines, fold in on the rest, and check where the held-out wine lands in the ranking over the full catalog. Held-out wines landed at a **0.92 mean percentile (median 0.978, 70% inside the top 5%)**. A popularity echo would have produced percentiles centered near the held-out wine's popularity rank, uncorrelated with the individual user. It did not. The fold-in is genuinely personalized.

&nbsp;<br>

## Other Decisions Worth Recording

**Consulting a sommelier instead of tuning the content weights.** The wine content vector has five attribute blocks (acidity, body, region, abv, grape), and their relative weights had to come from somewhere. The default instinct is to sweep them — treat the weights as hyperparameters, grid-search on held-out ratings, keep the winner. We deliberately did not.

Instead we consulted **Nitsan Granot, sommelier at Claro restaurant in Tel Aviv**, and asked which attributes actually decide whether a wine matches a palate or a dish. Her answer was structural and emphatic: acidity first, because it is what cuts fat and lifts a dish; then body, which determines whether the wine overwhelms the food or is overwhelmed by it. Grape variety — the thing printed largest on the label and the first thing a non-expert reaches for — matters considerably less. We translated that into the shipped weights: acidity 0.368, body 0.368 (together $\sim 74\%$), region 0.158, abv 0.053, grape 0.053.

 Rating data does not directly measure pairing suitability, and X-Wines grape labels are noisy, so relying heavily on them would degrade quality. We prioritized structural characteristics (acidity and body make up ~74% of the weight) which is more robust. The weights are applied at run-time, allowing easy adjustments.

**Region Rollup**: X-Wines features 2,160 distinct wine appellations. This is too sparse for content similarity (two Burgundies from adjacent villages would share nothing). We wrote a rollup script to group these into 107 parent regions (e.g., Pauillac → Bordeaux), making region comparison a useful feature.

**Focusing on Wine over Beer**: We initially planned to support both wine and beer. We dropped beer to focus our development effort and because wine is a better fit. Wine pairing rules are structured and well-documented. Additionally, X-Wines provided structured, dense ratings and attributes, whereas we lacked a high-quality dataset for beer.

**Extracting Pairing Rules**: The Wine-Food pairing dataset (~35k rows) contains rule-based category scores rather than ingredient-level patterns. Instead of training a model on noise, we extracted the underlying lookup table directly. This rule table is blended with recipe category similarity at run-time.

**Denormalizing Ingredients**: Instead of joining a 2M+ row ingredients table, we store recipe ingredients as comma-separated strings. This speeds up candidate scoring from database queries to simple string parses.

**Deployment via Docker**: To let users try the app without downloading datasets and training models (which takes over an hour), we deployed the services on Render. The ~940MB of trained models and database files are published as release assets and downloaded during build.

**Testing**: We wrote 530+ backend tests and 63 end-to-end frontend tests. This coverage was essential after a scoring calibration bug occurred that did not raise exceptions but distorted recommendations.

&nbsp;<br>

## Open Issues & Limitations

**Manual Offline Retraining**: Collaborative filtering models are trained offline on static datasets. New in-app ratings are used at serve time, but updating the latent factors requires manually running the training scripts.

**Candidate Retrieval Latency**: SQLite indexing is used to filter 231k recipes down to ~200. This works for our scale but won't scale to millions of recipes.

**Noisy Grape Labels**: The grape labels in the X-Wines dataset contain errors. We minimized this issue by keeping the grape weight low (5%) in favor of structural attributes (74% acidity/body), but the raw tags remain noisy.

**Weak Positive Ratings in ALS**: Currently, all ratings (including 1-star) are treated as weak positives in the ALS confidence matrix. We have not yet tested excluding low ratings (e.g., keeping only ratings $\ge 4$) to see if it improves recommendation accuracy.

&nbsp;<br>

## Future Work

### Planned Fixes
**Automated Retraining**: Set up a scheduled job to automatically retrain the recipe and wine models with new in-app user ratings.

**Vector-Search Candidate Retrieval**: Move candidate generation from SQLite to a vector search engine (like FAISS or Qdrant) to support fast searches over larger datasets.

**Clean Wine Metadata**: Cross-reference X-Wines with external APIs to correct erroneous grape labels.

**Filter Out Low Ratings in CF**: Build the ALS confidence matrix using only ratings $\ge 4$ to test if ignoring disliked wines improves NDCG@10.

### Product Extensions

**Supermarket Integration**: Sync with grocery store purchase history to automatically populate the user's pantry with accurate purchase dates and items.

**Israeli Wine Catalog**: Add local Israeli wines with store availability and pricing to make recommendations more practical for local users.

**Expand Beverages**: Add beer and non-alcoholic drinks. The pairing engine already maps foods to a 12-dimensional category space, making it easy to map new drinks.
