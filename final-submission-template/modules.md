# Modules Description

## Product Overview & Value Proposition (Product POV)

### The Real-World Problem Solved
Food waste is a major financial, logistical, and environmental challenge in modern households. Pantries and fridges are full but invisible — consumers frequently purchase ingredients, put them away, and forget they exist until they have already expired. Even when users notice an item approaching its expiration date, they struggle to spontaneously recall a recipe that utilizes it, resulting in preventable food waste and unnecessary grocery expenditures.

Traditional recommender systems and recipe platforms ask: *"What sounds good to eat?"* or *"What do you feel like cooking?"*  
**eXpairing re-engineers this paradigm.** eXpairing operates as an intelligent culinary assistant that asks:
> *"Given what is expiring in your fridge right now and how you actually cook, what is the highest-quality meal you will thoroughly enjoy?"*

### Key Product Value Drivers
1. **Dual-Signal Balancing (Preference vs. Feasibility)**: Combines machine learning preference predictions (what you enjoy) with physical household constraints (what is expiring and available in your fridge right now).
2. **Seamless Multi-Modal Inventory Ingestion**: Multi-modal vision scanning (GPT-4o / Gemini 2.5 Flash) and debounced ingredient autocomplete eliminate the friction of manual pantry logging. An automated canonicalization engine strips store packaging noise ("Tnuva 3% Milk") to map items onto standardized recipe tokens ("milk").
3. **Behavioral Adaptation & Revealed Preference Learning**: Recognizes that users often exhibit aspirational bias when setting waste aversion preferences ($\beta$). eXpairing tracks revealed cooking behavior (`n_missing`) and smoothly drifts $\beta$ via Exponential Moving Average (EMA) batch updates, displaying a profile alert when stated and revealed preferences diverge by $>10\%$.
4. **End-to-End Culinary & Dining Experience**: Extends beyond recipe recommendation into a full household utility. Includes interactive step-by-step cooking instructions, persistent shopping list management with one-click check-offs, 7-day skip exclusion memory, and a dedicated **Wine Recommender** that pairs wines directly with recipes or suggests personalized bottles matching user palates.

&nbsp;<br>

## Datasets Used

- **Food.com Recipes and User Interactions** — Primary dataset for the recipe recommender, [Kaggle Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). Contains 231,637 recipes, 1,132,367 user reviews/ratings (1–5 explicit stars), ingredient tokens, tags, cooking minutes, and step-by-step instructions.
- **X-Wines Dataset (Full)** — Primary dataset for the wine recommender, [GitHub Repository](https://github.com/rogerioxavier/X-Wines). Contains 100,646 wines and 21 million user ratings, along with structured metadata (grapes, 2,160 appellations, body, acidity, alcohol content).
- **Wine and Food Pairing Dataset** — Training input for the recipe-wine pairing engine, [Kaggle Dataset](https://www.kaggle.com/datasets/wafaaelhusseini/wine-and-food-pairing-dataset). ~35K rows, each labeling a (wine category × food category) combination with a 1-5 pairing-quality score. Signal analysis (`data/pairing/check_ingredient_signal.py`) showed the labels are category-level rule-generated, so we extract the underlying rule table directly — per-cell mean quality with injected contrast rows dropped (`data/pairing/extract_pairing_rules.py` → `models/pairing_rules.json`) — rather than fit a model to ingredient-level noise.

Additional data-related information:
- **Canonicalization & Entity Resolution**: Engineered a rule-based canonicalizer (`backend/canonicalizer/ingredient_map.py`) and integrated OpenFoodFacts API to resolve real-world store item names into Food.com canonical vocabulary tokens (e.g. mapping brand variants like "Tnuva 3% Milk" to canonical "milk", and synonyms like "aubergine" to "eggplant").
- **Region Rollup (Wine Domain)**: Developed `data/wine/region_rollup.py` to collapse 2,160 raw wine appellations down to 107 parent regions (e.g. Pauillac → Bordeaux, Meursault → Burgundy), enabling meaningful content-based region overlap.
- **Multi-Modal AI Extraction**: Integrated GPT-4o Vision and Gemini 2.5 Flash to parse physical fridge photographs, extract item lists with expiration dates, and insert them directly into our database schema.

&nbsp;<br>

## Data Preprocessing & Data Science Decisions

Raw data is treated as immutable: no cleaning script ever writes back over a source file. Each pipeline reads raw inputs and emits separate clean artifacts (`data/wine/clean_wines.py` → `clean_wines.csv` + `clean_ratings.csv`), so every transformation is reproducible from scratch and a bad cleaning rule can never destroy the original dataset.

### Wine Data Cleaning (`data/wine/clean_wines.py`)
- **Streaming ingestion with explicit dtypes**: The 21M-row ratings file is ~1GB on disk and would expand to 3–5GB if loaded whole. It is streamed in 500k-row chunks with `dtype` pinned to `int32`/`float32` rather than letting pandas infer `int64`/`float64` — halving memory and preventing silent type-inference errors.
- **Safe parsing of stringified lists**: X-Wines stores list columns (`Harmonize`, `Grapes`, `Vintages`) as literal Python syntax (`"['Beef', 'Lamb']"`). These are parsed with `ast.literal_eval`, which evaluates only literals, rather than `eval`, which would execute arbitrary code from a data file. Malformed values degrade to an empty list instead of crashing the pipeline.
- **Rating validation and clamping**: Rows with null ratings are dropped (they cannot be trained on) and ratings are constrained to the valid $[1, 5]$ scale — anything outside is a data error, not a signal.
- **Sparsity filtering**: Users with $< 5$ ratings and wines with $< 5$ ratings are dropped (`MIN_RATINGS_PER_USER = MIN_RATINGS_PER_ITEM = 5`). The wine rating median is 11, so this trims the cold tail that cannot produce a stable latent factor. Profiling showed X-Wines is already pre-filtered to $\ge 5$, but the rule is stated explicitly so it is visible and adjustable rather than an invisible property of the source.
- **Column pruning**: Columns unused by any model or endpoint (`Website` with 18k nulls, `Elaborate`, `Code`, `RegionID`, `WineryID`, `Vintages`) are dropped, and the survivors renamed to canonical schema names.
- **Post-clean assertions**: Cleaning code has bugs like any other code, so the pipeline asserts its invariants before anything reaches the database — wine IDs unique and non-null, ratings inside $[1,5]$, no null user/wine IDs, `harmonize_csv` populated for every wine (its loss would silently break food pairing), and zero orphan ratings referencing wines absent from the catalog.

### Feature Engineering Decisions
- **Region rollup** (`data/wine/region_rollup.py`): Raw region one-hot is near-useless — 2,160 distinct values, the top 50 covering only $\sim 42\%$ of wines, and two wines almost never sharing an *exact* appellation. Each `(region, country)` pair resolves through three tiers: an explicit leaf→parent map for well-known appellations (Pauillac → Bordeaux, Meursault → Burgundy, Napa Valley → California); keyword/substring rules catching un-enumerated sub-appellations (the various "… Grand Cru" variants); and finally a fallback to the country name — coarse, but an honest default for the obscure long tail rather than a fabricated match. The result is a static committed artifact (`region_rollup.json`): pure lookup, no ML, no network.
- **ABV clipping**: Raw `abv` values range across a dirty $0..50$. Values are clipped to a $[5.0, 16.0]$ band and min-max scaled into $[0,1]$, so a single corrupt outlier cannot dominate the vector's alcohol dimension. Missing values default to 13.0.
- **Ordinal encoding of palate attributes**: `acidity` (Low/Medium/High) and `body` (Very light-bodied … Full-bodied) are mapped onto ordered numeric scales in $[0,1]$ rather than one-hot encoded, because these attributes are genuinely ordered — a medium-bodied wine sits *between* light and full, and one-hot encoding would discard that.
- **TF-IDF configuration (recipes)**: `ngram_range=(1,2)` so both single ingredients ("garlic") and compound ones are captured; `min_df=2` drops hapax ingredients appearing in a single recipe; `max_features=20_000` caps vocabulary for memory; `sublinear_tf=True` applies $\log(1 + tf)$ so a recipe listing an ingredient repeatedly does not dominate.
- **Mean-centering before item-item similarity**: The recipe item-item matrix (`backend/ml/item_similarity.py`) subtracts each user's mean rating before computing cosine similarity, removing per-user rating-scale bias (the generous rater who never scores below 4). Recipes with $< 5$ ratings are excluded as too sparse to yield a meaningful row, and only the top $K=50$ neighbours per recipe are retained for memory.
- **Recipe CF user filtering**: `train_cf.py` excludes users with $< 3$ ratings from Funk SVD training, and augments the explicit rating set with synthetic ratings derived from cook events, since cooking a recipe is weaker evidence than an explicit 5-star rating but far from no evidence.

### Frozen Evaluation Split (`backend/ml/wine/training/build_wine_split.py`)
Trustworthy comparison between models requires that they be measured on identical data, so the wine train/test split is built **once** and frozen to disk (`models/wine_split/`: `train.npz`, `test.npz`, `user_ids.npy`, `item_ids.npy`, `split_meta.json`). Every subsequent experiment — the ALS-vs-SVD bake-off, the alpha sweep, the factors/regularization grid, the popularity baseline — loads that exact split.

This replaced an earlier approach that used a random `train_test_split`, which had two defects:
- **Temporal leakage**: scattering each user's ratings randomly lets the model "see the future", using a late rating to predict an earlier one. Production recommendation always predicts forward.
- **Unreproducible noise**: a fresh random draw per run means a tuning gain smaller than the split's own variance is indistinguishable from luck.

The frozen split uses **leave-$k$-out** (`k=5`), holding out exactly $k$ items per eligible user (those with $> k+1$ ratings) — the standard recsys protocol — over 16.2M train / 4.4M test ratings.

### Building the Shared Food-Category Space (Pairing Pipeline)
Wines and recipes can only be compared if they occupy the same vector space. Three data sources speak three different vocabularies, and reconciling them is the core preprocessing problem of the pairing engine. It is solved in three staged modules, none of which trains a model.

1. **Shared vocabulary** (`data/pairing/pairing_vocabulary.py`): The 12 canonical food categories are adopted from the labeled pairing dataset's `food_category` column, because the pairing *rules* and test labels are already expressed in exactly those terms. The wines' 66 `harmonize_csv` tokens are mapped onto them. Two decisions matter here. A token may carry **more than one** category (Lasagna is both Red Meat and Cheese), so the map's values are lists. And tokens describing a *course or dish type* rather than a sensory class — "Aperitif", "Appetizer", "Salad", "Pasta", "Pizza" — map to `None` and are deliberately **not** forced into a category: honest coverage beats fake precision.
2. **Wine → category vector** (`data/pairing/build_wine_pairing_vectors.py`): A wine's harmonize tokens become a **presence set**, not counts. Listing both Beef and Lamb does not make a wine "more Red Meat" than one listing Beef alone, so the category is binary-present and the resulting 12-dim vector is L2-normalized.
3. **Recipe → category vector** (`data/pairing/recipe_categories.py`): Food.com has ~15k distinct ingredient strings with endless variants ("garlic", "garlic cloves", "garlic powder", "minced garlic"), so exact matching would miss nearly everything. A hand-written **signal lexicon** is matched as whole words inside each ingredient — `"salmon"` matches "smoked salmon" and "salmon fillet", while `"ham"` is word-boundary guarded so it does not match "graham". Ingredients that appear in everything and signal nothing (salt, butter, water, flour, sugar) are simply absent from the lexicon and contribute zero. A recipe receives a **weighted set** of categories rather than a single label — lobster bisque is both Seafood and Creamy — L2-normalized to match the wine convention. Recipes hitting no signal keyword fall back to Vegetarian.

**Why hand-written rules instead of a learned model.** `data/pairing/check_ingredient_signal.py` asks a precise question: for a fixed (wine, food_category) pair, does `pairing_quality` vary by the specific `food_item`, or is it constant within the category? If quality is fully determined by the category, then training an ingredient→wine model can only *re-learn the category map* — a lossy copy of a rule we can simply write down. The analysis showed the labels are category-level rule-generated, with characteristic per-cell base quality plus noise, plus deliberately injected "contrast" rows (forced 1s and 5s). So `extract_pairing_rules.py` drops the contrast rows and reads the rulebook straight out of the data as per-cell means (`models/pairing_rules.json`, with a `global_mean` fallback for unseen cells). Checking whether a dataset contains the signal you intend to learn is cheaper than training on it and then wondering why the model generalizes poorly.

### Sommelier Weighting — Giving Extra Credit to Palate Structure
The wine content vector is built from five attribute blocks, but they are deliberately **not** weighted equally. The relative importance of each block was not tuned statistically — it was **elicited from a domain expert**. We consulted **Nitsan Granot, sommelier at Claro restaurant in Tel Aviv**, on which wine attributes actually drive a pairing or a palate match, and translated her account into the block weights below. Her guidance was unambiguous: acidity and body carry a wine, and grape variety matters far less than the label suggests.

The matrix is stored **unweighted** at rest (`models/wine_cb_matrix.npz`, each block independently normalized), and these weights are applied at **serve time** so they can be retuned — or overridden per request — without retraining anything.

| Block | Weight | Rationale |
| --- | --- | --- |
| Acidity | 0.368 | Palate-first: structural attributes are what a sommelier actually matches on. |
| Body | 0.368 | Together with acidity, $\sim 74\%$ of the vector. |
| Region | 0.158 | Meaningful only after the rollup to 107 parent regions. |
| ABV | 0.053 | Weak standalone signal; largely correlated with body. |
| Grape | 0.053 | Deliberately small — X-Wines grape labels are noisy. |

Acidity and body together carry $\sim 74\%$ of the weight. This encodes a **domain prior in place of a statistical one**: per Granot, how a wine *feels* — its weight and its cut — governs what it pairs with far more than which grape is printed on the label. Acidity is the attribute a sommelier reaches for first, because it is what cuts fat and lifts a dish; body determines whether a wine is overwhelmed by the food or overwhelms it.

The expert weighting also proved **defensively correct on data-quality grounds**, for a reason unrelated to why it was chosen. X-Wines grape tags are unreliable (a Cabernet blend tagged "Pinot Noir"), so holding the grape block at $\sim 5\%$ bounds how far that label noise can propagate into the recommendations. A purely statistical fit over this data might well have leaned harder on the grape block and inherited its noise.

### Diversity Reranking (MMR) Across All Three Ranked Surfaces
Relevance alone produces monotony: the top-scoring items for a given palate are frequently near-duplicates of one another. All three ranked surfaces therefore apply **Maximal Marginal Relevance**, greedily selecting the candidate maximizing $\text{MMR}(i) = \lambda \cdot \text{score}(i) - (1 - \lambda) \cdot \max_{j \in \text{selected}} \text{sim}(i, j)$. The highest-scored item is always picked first; each subsequent pick is penalized for similarity to what is already selected.

- **Recipe feed** (`GET /recipes/ranked`) — $\lambda = 0.7$, similarity from ingredient Jaccard, over the top 60 candidates.
- **Wine feed** (`GET /wine/ranked`) — $\lambda = 0.7$ (the shared `MMR_LAMBDA` default in `backend/services/wine/helpers.py`), similarity from pairwise content-vector cosine, over a pool of the top $3 \times \text{top\_n}$ by score. Applied in every user state, including cold start, so a first impression is not $N$ near-identical bottles.
- **Recipe-wine pairing** (`POST /wine/pair`) — $\lambda = 0.8$, the one surface that deviates. Relevance is held more strongly primary here because a pairing answers a specific question about a specific dish: the user wants bottles that genuinely match the recipe, and is less served by variety purchased at the cost of pairing quality.

&nbsp;<br>

## Technologies and Frameworks

### Frontend
- **React 18 & TypeScript** — component-based single-page application with static type safety.
- **Vite** — frontend build tool and hot-reloading development server.
- **CSS Modules & Custom Properties** — custom styling, score ring graphics, and responsive layouts.
- **Axios** — HTTP client for asynchronous REST API communication (`client.ts` and `wine.ts`).
- **Lucide React** — icon set used across the SPA screens.
- **Playwright** — end-to-end browser testing and automated presentation recording (63 tests).

### Backend
- **FastAPI** — high-performance Python web framework for REST API endpoints.
- **Pydantic** — data validation and settings management using Python type annotations.
- **SQLAlchemy** — ORM for database abstraction and transaction handling.
- **Uvicorn** — ASGI web server implementation.

### Algorithmic & ML
- **Scikit-surprise** — matrix factorization training (Biased Funk SVD via Stochastic Gradient Descent for recipes).
- **Implicit** — confidence-weighted Alternating Least Squares (ALS) matrix factorization for wine recommendations.
- **Scikit-learn** — TF-IDF vectorization (`TfidfVectorizer`) and vector cosine similarity.
- **SciPy & NumPy** — sparse matrix operations (`csr_matrix`), chunked cosine similarity, and array manipulations.
- **RapidFuzz** — fuzzy string matching for ingredient match ratios and canonical mapping.

### Data Platforms
- **SQLite** — relational database for persistent storage of user profiles, active pantry items, events, wine stats, and shopping lists.
- **Numpy/Pickle/JSON Files (`.npz`, `.pkl`, `.json`)** — serialized static model storage for offline-trained CF matrices, TF-IDF weights, and region mappings.

### AI & Computer Vision
- **OpenAI GPT-4o API (`OPENAI_API_KEY`)** — primary multi-modal vision model for fridge photo scanning, product label detection, and printed expiration date extraction.
- **Google Gemini 2.5 Flash API (`GEMINI_API_KEY`)** — alternative supported vision provider using structured JSON schema output (`response_mime_type='application/json'`).
- **OpenFoodFacts API** — external product lookup supporting brand-name resolution during canonicalization.
- **Deterministic Mock Scanner** — dev/demo fallback allowing complete offline execution without requiring paid API tokens.

&nbsp;<br>

## Main Algorithms & Core Rationale

A summary of the primary algorithms developed across both Recipe and Wine domains, highlighting what each algorithm does and why it was ideal for our system:

### 1. Recipe Collaborative Filtering — Biased Funk SVD (Matrix Factorization)
- **What it does**: Predicts personalized user ratings for recipes ($\ge 5$ ratings) by learning 50 latent preference factors per user and recipe, while explicitly accounting for user rating scale bias ($b_u$) and recipe popularity bias ($b_r$) via $\mu + b_u + b_r + p_u \cdot q_r^T$.
- **Why it was ideal**: Designed specifically for highly sparse explicit rating matrices (~99.998% empty). Training on observed ratings via Stochastic Gradient Descent (SGD) avoided the heavy noise and computational burden of dense matrix imputation.
- **Serving detail**: Cook events with no explicit star rating are converted into synthetic ratings (`max(3.0, 4.0 - n_missing*0.3)`) and folded into the training data. Warm-user predictions load from the offline artifact `cf_model.pkl`.

### 2. Recipe Cold-Start CF — Preference-Seeded Item Similarity
- **What it does**: Evaluates cold-start users ($< 5$ ratings) by mapping user dietary tags and pantry items onto a pre-computed sparse item-item cosine similarity matrix (`item_sim_matrix.npz`) built from mean-centered co-rating patterns across 51k+ recipes.
- **Why it was ideal**: Solves user cold-start without relying on unpopulated user vectors or generic, unpersonalized popularity lists. Using pantry items as preference anchors allows content signals to bootstrap recommendations directly into the behavioral item graph space.
- **Serving detail**: Between 1 and 4 ratings, the two CF strategies are blended via `alpha = n_ratings / 5`, giving a continuous cold-to-warm transition rather than a hard switch.

### 3. Recipe Content-Based Engine — TF-IDF Profiling
- **What it does**: Represents recipe ingredient lists as 20,000-dimensional TF-IDF vectors (unigrams + bigrams, stored in `cb_matrix.npz`) and computes cosine similarity against user taste profile vectors (built from pantry contents for cold users, or rating-weighted historical recipe vectors for warm users).
- **Why it was ideal**: Automatically highlights distinctive ingredients (saffron) while discounting common staples (salt), and captures cuisine affinities (miso + soy sauce matching Japanese dishes) without any explicit cuisine tag metadata. Exact lexical matching over canonical tokens is superior to dense text embeddings in an inventory system, as having butter in the fridge should prioritize actual butter recipes rather than semantic substitutes like margarine.

### 4. Wine Collaborative Filtering — Confidence-Weighted ALS
- **What it does**: Ranks wines for personalized feeds ("Suggest me a wine") using confidence-weighted Alternating Least Squares ($C = 1 + 5\cdot\text{rating}$, factors=64, reg=0.05) trained on 21M X-Wines ratings. Solves online user factor updates at runtime for active app users.
- **Why it was ideal**: Wine data consists of explicit ratings without implicit interaction logs. ALS treats unobserved items as confidence-weighted weak negatives, making it natively suited for ranking un-rated items (achieving an NDCG@10 of 0.0291, 4× popularity floor).

### 5. Wine Content-Based Model — Sommelier Structured Vectors
- **What it does**: Constructs content vectors across structured wine attributes (grape multi-hot, parent region rollup, body, acidity, abv) weighted by sommelier palate-first priors where structural attributes (body + acidity $\sim 74\%$ weight) guide cosine similarity.
- **Why it was ideal**: X-Wines contains zero free-text reviews, making NLP embeddings inapplicable. A structured vector with sommelier-weighted structural blocks provides highly interpretable attribute matching.

### 6. Automated Recipe-Wine Pairing Engine — 12-Dim Category Vectors & Empirical Rules
- **What it does**: Automatically pairs wines with recipes (`POST /wine/pair`) by projecting recipe ingredients onto a 12-dimensional food category vector space (`recipe_categories.py`) and blending category cosine similarity (`ALPHA_COSINE=0.6`) with empirical sommelier pairing rules (`BETA_RULES=0.4`) extracted from labeled pairing matrices (`pairing_rules.json`), reading pre-computed wine vectors from `models/wine_pair_matrix.npz`.
- **Why it was ideal**: Bridges the cross-domain gap between food ingredients and wine styles by aligning both items in a shared 12-category culinary space.

### 7. Domain Constraints & Diversity Reranking
- **Expiry Urgency & Match Penalization**: Adjusts rankings using exponential expiry decay ($\exp(-k \cdot \text{days})$, normalized by pantry size) and per-user ingredient availability match ratios ($\beta$) computed via RapidFuzz fuzzy string matching (threshold 75), prioritizing waste reduction without overriding base preferences.
- **Maximal Marginal Relevance (MMR)**: Greedily reranks top candidate pools to balance relevance against diversity. The recipe feed ($\lambda = 0.7$, ingredient Jaccard) and the wine feed ($\lambda = 0.7$, content-vector cosine) share the same trade-off; recipe-wine pairing raises it to $\lambda = 0.8$, holding relevance more strongly primary because a pairing answers a specific question about a specific dish.

### 8. Revealed Preference Learning — EMA Beta Drift
- **What it does**: Tracks the number of missing ingredients (`n_missing`) each time a user cooks, derives a revealed waste-aversion preference, and drifts the stored $\beta$ via a daily batch EMA update (`new_β = 0.85·current_β + 0.15·revealed_β`).
- **Why it was ideal**: Users state aspirational preferences that their cooking behavior contradicts. EMA smoothing adapts the weight without letting a single unusual cook event swing the feed, and the system raises a profile warning when stated and revealed $\beta$ diverge by $>10\%$.

&nbsp;<br>

## Database Schema & Data Layer Structure

The persistent storage layer is implemented in SQLite via SQLAlchemy ORM (`backend/db/models.py`), organized into 7 tables across both domains to balance normalization and real-time query performance. Foreign keys and cascades are enforced at the ORM layer.

```
┌───────────────┐       1:N       ┌───────────────────┐
│     users     ├────────────────►│   pantry_items    │
└───────┬───────┘                 └───────────────────┘
        │ 1:N
        ├────────────────────────►┌───────────────────┐
        │                         │shopping_list_items│
        │ 1:N                     └───────────────────┘
        ▼
┌───────────────┐       N:1       ┌───────────────────┐
│  user_events  │◄────────────────┤      recipes      │
└───────────────┘                 └───────────────────┘

┌───────────────┐       N:1       ┌───────────────────┐
│  wine_events  │◄────────────────┤       wines       │
└───────────────┘                 └───────────────────┘
```

### Table Definitions & Architectural Design:
1. **`users`**: Stores core profile records including auto-increment primary key `id`, display `name`, stated waste-aversion `beta` (default 0.35), comma-separated `diet_tags` (e.g., `"vegetarian,gluten-free"`), and model flags (`has_cf`, `has_cb`).
2. **`pantry_items`**: Manages current fridge inventory. Stores `user_id` (FK → `users.id`), canonical `ingredient` name (e.g., `"milk"`), original `raw_name` from vision/manual scans (e.g., `"Tnuva 3% Milk"`), `expiry_date`, and `quantity`. Canonicalization bridges real-world brand labels to model vocabularies.
3. **`recipes`**: Read-only reference table seeded from Food.com (231,637 rows). Stores `id`, `name`, `ingredients_csv`, `tags_csv`, `minutes`, `n_steps`, `avg_rating`, `n_ratings`, `description`, and `steps_json`.
   - *Denormalization Rationale*: Rather than creating a joined `recipe_ingredients` mapping table across 2M+ rows, `ingredients_csv` stores comma-separated canonical strings. This intentional denormalization allows $O(1)$ string parsing per candidate during real-time candidate scoring.
4. **`user_events`**: Central event stream logging all user actions. Stores `user_id` (FK → `users.id`), `recipe_id` (FK → `recipes.id`), `event_type` (`"cook"`, `"skip"`, `"rate"`), star `rating` (1–5), and `n_missing` ingredients at cook time.
   - *Single Table Discriminator Design*: Consolidating explicit ratings, implicit cook events, and skip exclusions into one unified event table simplifies analytical joins and event stream processing.
5. **`shopping_list_items`**: Persistent buy-list storing `user_id` (FK → `users.id`), `ingredient`, `source_recipe_id`, denormalized `source_recipe_name` for UI display, and `is_checked` boolean.
6. **`wines`**: Flat catalog table for the wine module (100,646 rows from X-Wines). Stores `id`, `winename`, `wine_type`/`style`, `vintage`, `abv`, `acidity`, `body`, `country`, `region`, `avg_rating`, and `n_ratings`.
7. **`wine_events`**: Event logging table for the wine module. Stores `user_id`, `wine_id`, `event_type` (`"rate"`), and star `rating` (1–5).

&nbsp;<br>

## System Architecture

The system is designed with a decoupled three-layer architecture: a React TypeScript frontend, a FastAPI Python backend, and an offline ML model storage layer backed by an SQLite database.

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser  (React + TypeScript)                                  │
│  Onboarding · Pantry · Feed · Detail · Wine · Profile · List    │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP / REST (Axios)
┌──────────────────────▼──────────────────────────────────────────┐
│  FastAPI  (Python)                                              │
│  /pantry · /recipes/ranked · /vision · /wine/ranked · /shopping │
└────┬───────────┬──────────┬──────────────┬──────────────────────┘
     │           │          │              │
     ▼           ▼          ▼              ▼
 SQLite DB    Scoring    Vision Agent   Beta Updater
 (SQLAlchemy) Service    (GPT-4o/Gemini)(Daily EMA Batch)
```

**Architectural Separation — What Lives Where:**
- **Database (SQLite)**: Stores dynamic, user-and-session-specific state — pantry items, expiry dates, event history (cook, rate, skip), shopping lists, and the per-user `β` preference parameter.
- **Model Artifacts (`models/`)**: Stores static, offline-trained community knowledge — matrix factorization weights (35k users × 231k recipes), sparse item-item similarity matrices, TF-IDF matrices, and wine ALS factors.
- **Serving Layer Runtime Fusion**: On every request, the system fuses static community taste knowledge from model files with live user state from the database.

**System Flow (Request Lifecycle for `GET /recipes/ranked`):**
1. **User Request & Context Retrieval**: Frontend calls `/recipes/ranked?user_id=X`. Backend retrieves user pantry items, expiry dates, dietary tags, and rating counts from SQLite.
2. **Candidate Generation (Stage 1)**: Filters 231,000 recipes down to ~200 candidates using a Bayesian average rating quality score (blending average rating with vote volume so a two 5-star review recipe does not unfairly beat a thousand 4-star review recipe). Skips recipes dismissed in the last 7 days and filters dietary conflicts.
3. **Multi-Model Scoring (Stage 2)**: For each candidate, evaluates four distinct signals, combined by the ranking formula `final_score = γ·cf_score + δ·cb_score + α·expiry_urgency + β·match_ratio`:
   - *Collaborative Filtering Score*: Automatically selects item-based CF (cold users 0 ratings) or biased Funk SVD matrix factorization (warm users ≥5 ratings). Between 1 and 4 ratings, blends both via `alpha = n_ratings / 5` to ensure smooth transitions.
   - *Content-Based Score*: Cosine similarity between candidate ingredient TF-IDF vector and user profile vector.
   - *Expiry Urgency Score*: Exponential decay calculation based on pantry items expiring soonest.
   - *Ingredient Match Ratio*: Fuzzy overlap between recipe requirements and available pantry items.
4. **Min-Max Calibration & Weighted Blending**: Each score component is min-max normalized across the candidate pool before blending. This is critical because raw CF scores cluster in a narrow range (0.30–0.38) while expiry urgency spans broadly (0.02–0.95); calibration ensures assigned weights (35% CF, 35% expiry, 20% match, 10% CB) maintain true proportions.
5. **MMR Reranking & Feedback Loop**: Top 60 candidates pass through MMR diversity reranking (λ=0.7) using ingredient Jaccard similarity. Top 20 recipes returned to UI with complete score breakdowns. User actions (`cook`, `rate`, `skip`) are saved to SQLite, feeding synthetic rating generators (`max(3.0, 4.0 - n_missing*0.3)`) and daily preference updates (`β`).

**System Flow (Request Lifecycle for `GET /wine/ranked` & `POST /wine/pair`):**
1. **Personalized Wine Feed (`GET /wine/ranked`)**: Applies hard style filters and checks user rating counts. *Cold start* (0 ratings) → Bayesian popularity prior. *Warming* (1-4 ratings) → `0.7·CB + 0.3·popularity` over a content-based taste profile. *Warm* (≥5 ratings) → `0.45·ALS_CF + 0.45·CB + 0.10·popularity` min-max calibrated. Active app users are folded in dynamically by solving online ALS user updates (`C = 1 + 5·rating`) against frozen item factors. In every state the top $3 \times \text{top\_n}$ pool is MMR-reranked (MMR $\lambda = 0.7$) over pairwise content-vector cosine.
2. **Automated Recipe-Wine Pairing (`POST /wine/pair`)**: Accepts a `recipe_id`, converts its ingredients to a 12-dim food category vector, and ranks wines by combining category cosine similarity (`ALPHA_COSINE=0.6`) with empirical sommelier pairing rules (`BETA_RULES=0.4`), then MMR-reranks the top pool at $\lambda = 0.8$ for bottle diversity.

&nbsp;<br>

## Modules & Source Code

eXpairing is organized into eleven modules spanning the **Recipe Recommender System** and the **Wine Recommender Module**. The table below maps each module to its technology stack, core responsibility, and source location; the algorithmic depth behind each engine is covered in [Main Algorithms & Core Rationale](#main-algorithms--core-rationale) above.

| Module | Technology | Core Responsibility | Source Code |
| --- | --- | --- | --- |
| **User Interface** | React 18, TypeScript, Vite, CSS Modules / Custom Properties, Axios, Lucide React | Renders the interactive SPA (Onboarding, Pantry, Ranked Recipe Feed, Recipe Detail with step-by-step instructions, Recipe Search/Browse, Wine Feed, Profile Settings, Shopping List). Collects explicit 1–5 star ratings and implicit cook/skip actions. Calls REST endpoints via Axios (`frontend/src/api/client.ts`, `frontend/src/api/wine.ts`) and renders score rings, match indicators, and component explainer bars. Supports client-side re-sorting of the loaded 20 recommendations by individual score component, plus wine style filters. | [`/frontend/`](../frontend/) |
| **API Gateway & Routers** | FastAPI, Python 3.9+, Pydantic, Uvicorn | Exposes the REST endpoints (`/recipes/ranked`, `/pantry`, `/events`, `/wine/ranked`, `/wine/pair`, `/wine-events`, `/shopping`, `/vision/scan`), validates JSON payloads, manages DB dependency sessions, and orchestrates call flows between data stores and serving engines. Sub-routers are split by domain: `pantry.py`, `recipes.py`, `users.py`, `vision.py`, `shopping.py`, `wine.py`. | [`/backend/routers/`](../backend/routers/) |
| **Database Schema & Data Models** | SQLite, SQLAlchemy ORM | Persists the 7 tables across both domains (`users`, `pantry_items`, `recipes`, `user_events`, `shopping_list_items`, `wines`, `wine_events`) and enforces foreign keys and cascades. Queried by routers and scoring services for user state, pantry contents, events, and shopping items. Denormalization and single-table-discriminator rationale documented in the schema section above. | [`/backend/db/models.py`](../backend/db/models.py) |
| **Recipe Recommendation & Scoring Engine** | Python, NumPy, SciPy | Implements the ranking formula `final_score = γ·cf_score + δ·cb_score + α·expiry_urgency + β·match_ratio`, min-max calibrates each component across the candidate pool so wide-ranging domain signals do not drown out narrow CF distributions, and applies MMR (λ=0.7) diversity reranking over the top 60 candidates. Reads pantry items, expiry dates, and diet tags from the DB; queries `serve_cf.py` and `serve_cb.py`; evaluates constraints from `expiry.py` and `ingredient_match.py`. | [`/backend/services/scoring.py`](../backend/services/scoring.py) |
| **Recipe Collaborative Filtering Engine (CF)** | Scikit-surprise (Biased Funk SVD / SGD Matrix Factorization), SciPy `csr_matrix` | Dual-strategy serving: latent-factor dot products for warm users (≥5 ratings) and item-item cosine similarity on mean-centered rating graphs for cold-start users (<5 ratings). Evaluated at request time by `scoring.py`; loads offline artifacts `cf_model.pkl` and `item_sim_matrix.npz`. Synthetic ratings from cook events feed training when explicit ratings are absent. | [`/backend/ml/serve_cf.py`](../backend/ml/serve_cf.py) & [`/backend/ml/cold_start.py`](../backend/ml/cold_start.py) |
| **Recipe Content-Based Filtering Engine (CB)** | Scikit-learn (`TfidfVectorizer`, `cosine_similarity`) | Encodes ingredient lists into 20,000-dimensional TF-IDF vectors (unigrams + bigrams) and maintains user taste profile vectors from pantry items (cold) or rating-weighted historical recipe vectors (warm). Supplies candidate-to-profile cosine similarity to `scoring.py`, reading pre-computed vectors from `cb_matrix.npz`. | [`/backend/ml/serve_cb.py`](../backend/ml/serve_cb.py) & [`/backend/ml/train_cb.py`](../backend/ml/train_cb.py) |
| **Wine Recommender Engine** | Implicit (Confidence-Weighted ALS), structured vector cosine matching | Powers "Suggest me a wine" by blending ALS collaborative filtering (21M X-Wines ratings), sommelier-weighted structured content matching (grape multi-hot, parent region rollup, body, acidity, abv), and a Bayesian popularity prior. Serves cold (0 ratings → popularity), warming (1–4 → `0.7·CB + 0.3·popularity`), and warm (≥5 → `0.45·CF + 0.45·CB + 0.10·popularity`, min-max calibrated) states via `/wine/ranked` and `/wine-events`, MMR-reranked at λ=0.7. Folds in active app users at runtime through an online ALS solver update. | [`/backend/routers/wine.py`](../backend/routers/wine.py), [`/backend/services/wine/scoring.py`](../backend/services/wine/scoring.py), & [`/backend/ml/wine/serving/`](../backend/ml/wine/serving/) |
| **Recipe-Wine Pairing Module** | Python, NumPy, SciPy sparse vectors | Serves `POST /wine/pair`: maps recipe ingredients onto a 12-dimensional food category vector (`recipe_categories.py`) and ranks wines by blending category cosine similarity (`ALPHA_COSINE=0.6`) with empirical sommelier rules (`BETA_RULES=0.4`, `pairing_rules.json`), reading `models/wine_pair_matrix.npz`. Applies MMR reranking (λ=0.8) for bottle diversity. | [`/backend/ml/wine/serving/serve_pairing.py`](../backend/ml/wine/serving/serve_pairing.py) & [`/backend/routers/wine.py`](../backend/routers/wine.py) |
| **User Profile Manager & Preference Learning (Beta Updater)** | Python, SQLAlchemy, Exponential Moving Average math | Manages persistent profile records (diet tags, stated `β`) and runs the daily batch preference update `new_β = 0.85·current_β + 0.15·revealed_β` from tracked `n_missing` counts. Consulted by scoring services on every request for per-user weighting; triggers a profile warning when stated and revealed `β` diverge by >10%. | [`/backend/services/beta_updater.py`](../backend/services/beta_updater.py) & [`/backend/db/models.py`](../backend/db/models.py) |
| **Expiry Urgency & Ingredient Matcher** | Python, RapidFuzz | Computes exponential expiry urgency (`exp(-k · days)`) normalized by pantry size, and performs fuzzy ingredient string matching (threshold 75) to produce availability ratios and missing-ingredient lists. Called directly by `scoring.py` during candidate ranking. | [`/backend/services/expiry.py`](../backend/services/expiry.py) & [`/backend/services/ingredient_match.py`](../backend/services/ingredient_match.py) |
| **Vision Agent & Ingredient Canonicalizer** | OpenAI GPT-4o API (`OPENAI_API_KEY`), Google Gemini 2.5 Flash API (`GEMINI_API_KEY`), RapidFuzz, OpenFoodFacts API | Processes fridge photos via multi-modal vision APIs to extract item labels, quantities, and printed expiry dates (`YYYY-MM-DD`), served at `/vision/scan` (or `/vision/mock` for API-key-free development). A two-step canonicalization pipeline strips brand names ("Tnuva", "Heinz") and noise adjectives, then fuzzy-matches against the 20,000-token recipe vocabulary (`FUZZY_THRESHOLD=70`) before insertion into the pantry tables. | [`/backend/services/vision_agent.py`](../backend/services/vision_agent.py) & [`/backend/canonicalizer/`](../backend/canonicalizer/) |
| **Persistent Shopping List Manager** | Python, SQLAlchemy, FastAPI | Manages the persistent buy-list over `/shopping/{user_id}`: adds missing ingredients from recipe detail cards, handles deduplication, item check-offs, clear-purchased actions, and source recipe attribution. | [`/backend/routers/shopping.py`](../backend/routers/shopping.py) |

&nbsp;<br>

## Development Environment
- **VS Code** - used for React frontend UI development, FastAPI backend service orchestration, and ML training scripts.
- **Pytest & Playwright** - used for backend unit testing (530+ tests) and end-to-end browser behavioral verification (63 tests).

&nbsp;<br>

## Evaluation

Recommendation quality was evaluated through offline metrics, algorithm bake-offs, lifecycle simulations, hyperparameter grid searches, and automated test coverage:

- **Algorithm Bake-Off (Wine CF - ALS vs Funk SVD)**: Evaluated on a frozen leave-5-out split (`models/wine_split/`, 16.2M train / 4.4M test ratings). Funk SVD achieved strong rating RMSE (0.596) but failed at ranking (NDCG@10 ~0.0006 ≈ random). Confidence-weighted ALS (alpha=5) achieved NDCG@10 of **0.0291** (4× popularity baseline of 0.0071), proving that for ranking tasks without explicit rating goals, ALS is the superior objective.
- **Wine ALS Hyperparameter Experiments** (all on the same frozen split): swept confidence scale alpha over {1, 5, 15, 40} — **alpha=5 won** (0.0291, +10% over the library-default 40, which over-saturated confidence); factors {64, 128, 200} × regularization {0.01, 0.05, 0.1} was flat at factors=64; alternative matrix weightings TF-IDF (~linear, noise-level difference) and BM25 (**collapsed, −75%** — default saturation far too aggressive for this data) confirmed linear alpha=5 as the practical pure-CF ceiling.
- **ALS Fold-In Validation (App Users)**: App users are not in the offline ALS factors, so serving folds them in by solving the ALS user update against frozen item factors. Validated via leave-one-out over 200 real users: held-out wines ranked at a **0.92 mean percentile** (median 0.978, 70% inside the top 5%) — confirming the fold-in produces genuinely personalized rankings, not popularity echoes.
- **Offline Recipe Rating Accuracy**: Evaluated on held-out Food.com test sets. Biased Funk SVD achieved an RMSE of **0.6136** (vs global mean RMSE baseline of 1.12), representing a 45% error reduction.
- **Cold-to-Warm Lifecycle Simulation**: Conducted simulated user lifecycle testing (`evaluate.py --lifecycle`) across interaction steps (0 to 10 ratings) to verify that the soft CF blend transition ramps smoothly without scoring discontinuities.
- **Weight Grid Search**: Executed grid search over weight combinations of γ (CF) and α (expiry) to validate default weights (γ=0.35, α=0.35, β=0.20, δ=0.10).
- **Test Suite Verification**: Maintained 530+ pytest unit and behavioral integration tests covering scoring math, decay rates, DB transactions, and API contracts, alongside 63 Playwright E2E tests.

&nbsp;<br>

## Main Features

- **Multi-Domain Hybrid Ranking Feeds**: Personalizes recipe and wine feeds with transparent score breakdown rings and explainer bars.
- **Automated Recipe-Wine Pairing**: Pair wines directly with any recipe via `POST /wine/pair`, using a hybrid of 12-dimensional category vector cosine similarity and empirical sommelier rule matrices.
- **Dynamic Feed Re-Sorting**: Client-side controls allow users to re-sort loaded recipe recommendations dynamically by individual components (e.g. sort strictly by expiry urgency or CF score).
- **Multi-Modal AI Vision Scanner with Dual Token Support & Canonicalization**: Users take photos of their fridge; GPT-4o (`OPENAI_API_KEY`) or Gemini 2.5 Flash (`GEMINI_API_KEY`) extracts ingredients and expiry dates. Raw packaging text is stripped of brand names ("Tnuva", "Danone", "Heinz") and fuzzy-matched onto the 20k Food.com canonical vocabulary.
- **Revealed Preference Learning (`β` Updater)**: Tracks missing ingredient counts (`n_missing`) when users cook. An EMA background process updates the user's `β` weight, adapting recommendations to match actual cooking habits over time.
- **Personalized Wine Module**: Provides tailored wine recommendations ("Suggest me a wine") with style selection chips, food pairing groupings, and star rating feedback.
- **Skip Exclusion Memory**: Suppresses skipped recipes from appearing in the user feed for 7 days to prevent recommendation repetition.
- **Persistent Shopping List Integration**: Allows users to add missing recipe ingredients directly to a persistent buy-list, complete with check-off mechanics and source recipe attribution.
