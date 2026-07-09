# Project Summary

This document records how exPairing was actually built: what we tried, what failed, what we concluded from each failure, and why the shipped version looks the way it does. Where the [Modules Description](modules.md) document describes the *final* system, this one describes the path to it — including the branches we abandoned.

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

This immediately created a new problem: matrix factorization is useless for a user with no ratings, and *every* new exPairing user has no ratings. The obvious fallback — serve a popularity list — was unacceptable, because a generic top-recipes list is exactly the non-personalized experience the product exists to replace.

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
One gap remained. The offline ALS factors are computed over X-Wines users; an exPairing app user does not exist in that factorization. At serving time we fold them in — solving the ALS user-factor update against the frozen item factors, using only their in-app ratings.

Fold-in is easy to get subtly wrong in a way that produces plausible-looking output: a poorly-conditioned solve tends to return something close to the global mean, which surfaces popular wines, which *look* like reasonable recommendations. We needed to prove the fold-in was producing genuine personalization rather than a popularity echo.

We validated with leave-one-out over 200 real app users: hold out one of the user's rated wines, fold in on the rest, and check where the held-out wine lands in the ranking over the full catalog. Held-out wines landed at a **0.92 mean percentile (median 0.978, 70% inside the top 5%)**. A popularity echo would have produced percentiles centered near the held-out wine's popularity rank, uncorrelated with the individual user. It did not. The fold-in is genuinely personalized.

&nbsp;<br>

## Other Decisions Worth Recording

**Consulting a sommelier instead of tuning the content weights.** The wine content vector has five attribute blocks (acidity, body, region, abv, grape), and their relative weights had to come from somewhere. The default instinct is to sweep them — treat the weights as hyperparameters, grid-search on held-out ratings, keep the winner. We deliberately did not.

Instead we consulted **Nitsan Granot, sommelier at Claro restaurant in Tel Aviv**, and asked which attributes actually decide whether a wine matches a palate or a dish. Her answer was structural and emphatic: acidity first, because it is what cuts fat and lifts a dish; then body, which determines whether the wine overwhelms the food or is overwhelmed by it. Grape variety — the thing printed largest on the label and the first thing a non-expert reaches for — matters considerably less. We translated that into the shipped weights: acidity 0.368, body 0.368 (together $\sim 74\%$), region 0.158, abv 0.053, grape 0.053.

Two things justify preferring the expert to the sweep. First, **the available objective was the wrong one**. A weight sweep would optimize agreement with X-Wines *ratings*, but the content model's job is palate and pairing similarity, which the rating data does not directly measure. Tuning against the metric we happened to have would have meant optimizing the wrong quantity precisely. Second — and we only appreciated this afterwards — **the grape labels in X-Wines are noisy**, so a statistical fit might well have leaned on the grape block and quietly absorbed that noise. The sommelier's palate-first prior holds grape at $\sim 5\%$ and bounds the damage, for reasons having nothing to do with data quality. The expert prior was the more principled choice, and by coincidence the more robust one.

We apply the weights **at serve time over an unweighted stored matrix**, precisely so this judgement stays contestable: they can be retuned, or overridden per request, without retraining anything.

**Region rollup over raw appellations.** X-Wines carries 2,160 distinct appellations. Used directly as a content-vector block, they are near-orthogonal — two Burgundies from adjacent villages share nothing, so the region signal contributes essentially zero similarity. We built `data/wine/region_rollup.py` to collapse them into 107 parent regions (Pauillac → Bordeaux, Meursault → Burgundy), which is what makes region overlap a usable content feature at all.

**Cutting beer to ship one beverage domain properly.** What is now the wine module was originally scoped in the plural — wine *and* beer, with a beer-wine relationship model connecting them. We cut beer, for two reasons that reinforced each other.

The practical reason was focus. Two drink domains meant two catalogs to source and clean, two cold-start policies, two content-vector designs, and a third cross-domain model on top of them — against a fixed deadline. Spread across all of that, each piece would have been a demo. Concentrated on one, wine became a module we could actually validate: a real bake-off against a frozen split, a hyperparameter sweep, and a leave-one-out check on the serving-time fold-in. We preferred one domain we had measured to two we had merely built.

The substantive reason is that wine is simply the better fit for this system, on both the data-science and the culinary side. Culinarily, wine and food share a vocabulary — body, acidity, tannin, weight, richness — and pairing is a mature, well-documented practice built on exactly those shared axes. That shared structure is what makes a cross-domain projection meaningful at all: our 12-dimensional food category space and the wine content vector line up because sommeliers had already established that they line up. Beer's descriptive axes (bitterness, malt profile, carbonation) map onto food far less directly, and beer-food pairing has a much thinner empirical rule base to draw on. On the data side, X-Wines gave us 21M explicit ratings across 100,646 wines plus structured attributes (grape, region, body, acidity, abv) — enough to train ALS meaningfully and to build an interpretable content vector without any free text. We had no beer dataset of comparable density or attribute richness, which meant a beer CF model would have been thin and the beer-wine relationship model would have been fitted on top of that thinness. The pairing engine works because the wine side is dense and the food side shares its structure; beer would have weakened both halves.

**Extracting the pairing rule table instead of fitting a model to it.** The Wine and Food Pairing dataset offers ~35K rows scoring (wine category × food category) combinations 1–5, and the obvious move is to train a model on it. Signal analysis (`data/pairing/check_ingredient_signal.py`) showed there was nothing to learn: the labels are category-level rule-generated, with no ingredient-level signal beneath them. Fitting a model would have meant fitting noise around a deterministic table. We extracted the table directly instead — per-cell mean quality, with injected contrast rows dropped (`data/pairing/extract_pairing_rules.py` → `models/pairing_rules.json`) — and blend it with category cosine similarity at `ALPHA_COSINE=0.6` / `BETA_RULES=0.4`. Checking whether a dataset contains the signal you intend to learn is cheaper than training on it and wondering why the model generalizes poorly.

**Denormalizing `ingredients_csv`.** A properly normalized `recipe_ingredients` join table would span 2M+ rows and require a join per candidate on every ranked request. Storing comma-separated canonical strings on the `recipes` row makes candidate scoring an $O(1)$ string parse. This is a deliberate trade of write-side elegance for read-side latency on the hot path.

**Deploying the application rather than shipping a clone-and-train.** Running exPairing from source means cloning the repository, downloading the Food.com and X-Wines datasets, seeding a database with 1.07M ratings, and training four models — the better part of an hour before the first recommendation appears. That is a reasonable thing to ask of a contributor and an unreasonable thing to ask of anyone who simply wants to see whether the system works. We deployed both services to a hosted platform so that the application can be opened at a URL. It was a convenience decision, and it turned into an instructive one.

The trained artifacts do not fit in the repository. `models/` and the seeded database total roughly 940MB, and individual files exceed GitHub's 100MB per-file limit, so both are gitignored. Rather than committing them through LFS or rebuilding them inside the image — which would mean a training run on every deploy — we publish them once as a release asset and have the Dockerfile fetch and unpack them at build time. This buys a fast, reproducible build at the cost of a new failure mode: the build argument carrying the artifact URL is easy to omit, and when it is omitted the fetch is skipped, the image ships with no models, the container starts, and the health check returns `200`. Only an actual recommendation request fails. We added assertions to the build that check for `fridge2fork.db`, `cf_model.pkl` and `wine_als_model.npz`, on the principle that a build which fails loudly is strictly better than a service that looks healthy and errors on first use. The frontend carries a symmetric hazard: the API URL is inlined into the JavaScript bundle at build time and falls back to `localhost` when absent, producing a page that renders perfectly and whose every request goes nowhere.

**Selecting an instance size was a measurement problem, not a budgeting one.** The backend holds four models resident and warms to roughly 825MB of process memory. The platform's smaller instances offer 512MB, and the way they fail is the point: the service starts, the health check passes, and the container is killed by the OOM handler on the first recommendation request. A deployment watched only through its health check would report itself healthy and serve nothing. Provisioning therefore has to be driven by resident memory *after every endpoint has been exercised*, not by memory at startup — the two differ by more than a factor of three here.

We measured rather than estimated. Running the image under hard memory caps (`docker run --memory=...`) established that 512MB dies on the first recommendation, and that under a 2GB cap fifty consecutive recipe-and-wine request pairs all return `200` with resident memory flat at ~890MB — no per-request leak. The same 2GB instance *is* killed under thirty-way concurrent load, and that told us something useful about what we were provisioning for. A person clicking through a feed issues sequential requests; sizing against a concurrency profile the application will never encounter would have meant paying for headroom against an imaginary load. We sized for the real usage profile and recorded the concurrency ceiling as a known bound rather than engineering it away.

We considered a cheaper path and rejected it. The recipe CF model accounts for 435MB of that resident total, yet the arrays its predictions actually read come to only ~101MB; the remainder is the pickled training set, which `predict()` never touches. Stripping it would have cut roughly 330MB and brought the service within reach of a smaller instance. The resulting footprint would have sat within a few tens of megabytes of the 512MB ceiling — close enough that any later change to the model or its dependencies would have silently reintroduced the OOM, and the symptom would have been a health check passing while every recommendation failed. Provisioning memory is reversible and instantly diagnosable; discovering at grading time that the margin was too thin is neither. The optimization is genuine and is recorded in *Future Work* rather than discarded.

The deployment also surfaced two defects that no test caught, because both were properties of the image rather than of the code. Purging the C compiler from the backend image to shrink it also removed a shared library that the ALS implementation loads at import; the container still started and `/health` still passed, and only a wine request would have revealed it. Separately, the frontend's nginx configuration substitutes the injected port at container start, and left unrestricted that substitution also consumed nginx's own internal variables and broke client-side routing on every deep link. Both were found by exercising the running containers endpoint by endpoint. Neither would have been found by a green build, and that is the same lesson the calibration bug taught in Milestone 4, arriving by a different route.

**Verification discipline.** The system carries 530+ pytest unit and behavioral integration tests over scoring math, decay rates, DB transactions, and API contracts, plus 63 Playwright end-to-end browser tests. The calibration bug in Milestone 4 was the reason: it was a scoring-math defect that produced no error, no exception, and a feed that looked superficially fine. Only a test asserting on the *relative contribution* of each component would have caught it, and after that we wrote tests at that level.

&nbsp;<br>

## Open Issues & Limitations

- **Offline CF Retraining Schedule**: Matrix factorization weights are trained offline on static Food.com and X-Wines ratings. In-app ratings and cook events are captured in SQLite and used at serving time, but folding them into the latent vectors requires triggering an offline retraining script by hand. The consequence is that community-level knowledge is frozen at training time; only per-user state is live.
- **Candidate Retrieval Is Not Sub-Millisecond**: Candidate generation uses database indexing and popularity caps in SQLite to narrow 231k recipes to ~200. This is adequate at our scale but is a linear-scan-shaped solution wearing an index, and it will not hold at millions of items.
- **Noisy Grape Labels (X-Wines)**: Grape variety tags in X-Wines are unreliable — a Cabernet blend may be tagged "Pinot Noir". This corrupts the grape block of the wine content vector and any grape-based UI text. The sommelier weighting deliberately keeps the grape block small ($\sim 5\%$) relative to the structural attributes (body + acidity, $\sim 74\%$), which bounds the damage; we chose palate-first weighting on sommelier grounds and it happened to also be the robust choice against this label noise. It is a mitigation, not a fix.
- **Untested CF Lever — Positive-Rating Cut**: The hyperparameter sweeps established α=5 linear weighting as the ceiling for *the current confidence matrix*, but that matrix includes every rating, so a 1-star wine still enters as a weak positive ($C = 6$ versus an unobserved cell's $C = 1$). Dropping ratings $< 4$ so that disliked wines stop acting as positives is the one identified pure-CF lever we did not get to test. It is the most likely route past the 0.0291 NDCG@10 ceiling.

&nbsp;<br>

## Future Work

### Closing the Known Gaps
These follow directly from the limitations above — each one is a specific, scoped fix to something we already know is wrong or missing.

- **Scheduled Background Retraining Pipeline**: Automate the offline retraining of both recipe and wine factorizations on a recurring schedule, ingesting accumulated in-app ratings and synthetic cook-event ratings, so community knowledge tracks the live user base rather than the seed dataset.
- **Vector-Search Candidate Retrieval**: Migrate candidate generation from SQLite indexing to a vector search engine such as FAISS or Qdrant, enabling sub-millisecond approximate similarity queries across millions of items and removing the ~200-candidate cap that currently bounds what the scoring stage can even consider.
- **Cleaner Grape Metadata**: Cross-reference X-Wines grape labels against an external wine reference source to repair mislabeled varieties, which would let the grape block carry more weight in the content vector without introducing noise.
- **Test the Positive-Rating Cut**: Rebuild the ALS confidence matrix over ratings $\ge 4$ only, re-run the α sweep on the same frozen split, and measure whether removing weak-positive noise from disliked wines moves NDCG@10 past 0.0291.

### Extending the Product
These are larger directions that would change what exPairing *is*, not merely how well it performs.

- **Supermarket Integration for Live Inventory Tracking**: Integrate with local supermarket systems (loyalty-card purchase history, online-order receipts, or store APIs) to track ingredient supplies automatically. This attacks the single biggest source of friction in the product: today the pantry is only as accurate as what the user scans or types. Purchase data would let the pantry populate itself at checkout, and would supply real expiry horizons from purchase dates rather than relying on the user reading a label. It also closes the shopping-list loop — the list could verify that a bought item actually arrived in the fridge.
- **Enriched Israeli Wine Coverage**: X-Wines is heavily weighted toward European and New World producers, so an Israeli user gets recommendations they cannot easily buy. Building out Israeli wine coverage — Galilee, Judean Hills, Negev producers — with local availability and pricing would make the wine module actionable for the market it was built in. This is a data-sourcing problem rather than a modeling one, and the content vector already generalizes: Israeli wines slot into the existing region rollup and sommelier-weighted attribute space without any change to the model.
- **Beer and Non-Alcoholic Beverages**: Restore the beverage breadth that was deliberately cut (see *Cutting beer to ship one beverage domain properly* above), extending the module to beer and to non-alcoholic options — sodas, juices, teas, alcohol-free wines and beers. Two things make this more tractable now than at the start. The pairing engine already projects both sides onto a shared 12-dimensional food-category space, so a new beverage domain needs only its own content vector and a mapping into those categories rather than a bespoke cross-domain model. And non-alcoholic coverage broadens the product's addressable audience considerably — to users who abstain for religious, medical, or personal reasons and are currently served nothing.
- **Filtering by Price, Calories, and Medical Constraints**: Add hard-constraint filters over both domains: price ceilings, calorie and macronutrient budgets, and medical or religious restrictions (diabetic sugar limits, sodium caps, allergen exclusion, kosher/halal certification). These are **constraints, not preferences**, and the distinction matters architecturally. The current dietary tags act as soft signals inside the candidate filter, but an allergen or a medical limit must never be traded off against a high CF score — no amount of predicted enjoyment justifies surfacing a dish that would harm the user. They therefore belong as hard filters applied *before* scoring, in the candidate-generation stage, rather than as another weighted term in the blend.
