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
The system uses a decoupled three-layer architecture:
1. **Frontend**: React SPA for user interaction and request handling.
2. **Backend**: FastAPI orchestrates scoring services, vision agents, and database operations.
3. **Data Layer**: SQLite stores state, while `models/` artifacts store static community knowledge.

**Request Lifecycle (e.g., `GET /recipes/ranked`)**:
1. Retrieve user context (pantry, expiry, diet).
2. Filter to ~200 candidates via Bayesian quality score.
3. Score candidates using hybrid signals (CF, CB, Expiry, Match Ratio).
4. Min-max normalize components and blend with calibrated weights.
5. Apply MMR reranking (λ=0.7) and return top 20 with explanation data.

&nbsp;<br>

## Modules & Source Code

eXpairing is organized into several modules. The table below maps each module to its technology stack, core responsibility, and source location:

| Module | Technology | Core Responsibility | Source Code |
| --- | --- | --- | --- |
| **User Interface** | React 18, TypeScript, Vite, CSS Modules / Custom Properties, Axios, Lucide React | React SPA showing Onboarding, Pantry, Feed, Recipe details, Wine Feed, and Shopping List. Handles user ratings, cooking logs, recipe skips, and frontend score breakdowns/filtering. | [`/frontend/`](../frontend/) |
| **API Gateway & Routers** | FastAPI, Python 3.9+, Pydantic, Uvicorn | FastAPI endpoints validating payloads, managing DB sessions, and calling backend services. | [`/backend/routers/`](../backend/routers/) |
| **Database Schema & Data Models** | SQLite, SQLAlchemy ORM | SQLite database with 7 tables mapping users, pantry items, recipes, events, and shopping lists. | [`/backend/db/models.py`](../backend/db/models.py) |
| **Recipe Recommendation & Scoring Engine** | Python, NumPy, SciPy | Ranks recipes using the hybrid formula, normalizes score components, and applies MMR diversity reranking. | [`/backend/services/scoring.py`](../backend/services/scoring.py) |
| **Recipe Collaborative Filtering Engine (CF)** | Scikit-surprise, SciPy `csr_matrix` | Serves Funk SVD ratings prediction for warm users and item-item similarity for cold-start users. | [`/backend/ml/serve_cf.py`](../backend/ml/serve_cf.py) & [`/backend/ml/cold_start.py`](../backend/ml/cold_start.py) |
| **Recipe Content-Based Filtering Engine (CB)** | Scikit-learn (`TfidfVectorizer`, `cosine_similarity`) | Computes cosine similarity between 20k TF-IDF recipe vectors and the user's taste profile. | [`/backend/ml/serve_cb.py`](../backend/ml/serve_cb.py) & [`/backend/ml/train_cb.py`](../backend/ml/train_cb.py) |
| **Wine Recommender Engine** | Implicit (Confidence-Weighted ALS), structured vector cosine matching | Ranks wines by combining ALS collaborative filtering, content similarity, and popularity. Folds in new ratings at runtime. | [`/backend/routers/wine.py`](../backend/routers/wine.py), [`/backend/services/wine/scoring.py`](../backend/services/wine/scoring.py), & [`/backend/ml/wine/serving/`](../backend/ml/wine/serving/) |
| **Recipe-Wine Pairing Module** | Python, NumPy, SciPy sparse vectors | Maps recipes onto a 12-dimensional food category space, ranks wines using cosine similarity and sommelier rules, and applies MMR. | [`/backend/ml/wine/serving/serve_pairing.py`](../backend/ml/wine/serving/serve_pairing.py) & [`/backend/routers/wine.py`](../backend/routers/wine.py) |
| **User Profile Manager & Preference Learning (Beta Updater)** | Python, SQLAlchemy, Exponential Moving Average math | Manages user profiles and runs a daily EMA job to adjust the waste-aversion weight (beta) based on actual cooking habits. | [`/backend/services/beta_updater.py`](../backend/services/beta_updater.py) & [`/backend/db/models.py`](../backend/db/models.py) |
| **Expiry Urgency & Ingredient Matcher** | Python, RapidFuzz | Computes exponential expiry decay and performs fuzzy ingredient matching to track missing ingredients. | [`/backend/services/expiry.py`](../backend/services/expiry.py) & [`/backend/services/ingredient_match.py`](../backend/services/ingredient_match.py) |
| **Vision Agent & Ingredient Canonicalizer** | OpenAI GPT-4o API, Google Gemini 2.5 Flash API, RapidFuzz, OpenFoodFacts API | Extracts ingredients and expiry dates from fridge photos using GPT-4o/Gemini. Cleans and maps raw names to recipe tokens. | [`/backend/services/vision_agent.py`](../backend/services/vision_agent.py) & [`/backend/canonicalizer/`](../backend/canonicalizer/) |
| **Persistent Shopping List Manager** | Python, SQLAlchemy, FastAPI | Handles shopping list CRUD operations, deduplication, check-offs, and recipe source tracking. | [`/backend/routers/shopping.py`](../backend/routers/shopping.py) |

&nbsp;<br>

## Evaluation

We evaluated recommendation quality and model performance using offline metrics, simulations, and testing:

- **Wine CF Bake-Off (ALS vs Funk SVD)**: Tested on a frozen leave-5-out split (16.2M train / 4.4M test ratings). While Funk SVD had a good rating RMSE (0.596), it failed at ranking (NDCG@10 ~0.0006). Confidence-weighted ALS reached NDCG@10 of **0.0291** (4x the popularity baseline of 0.0071), proving ALS is much better for ranking.
- **ALS Hyperparameter Tuning**: Sweeping parameters on the frozen split showed that `alpha=5` performed best (+10% over the default of 40, which oversaturated confidence). Matrix factors and regularization changes were flat beyond 64 factors. Alternate weightings like BM25 collapsed performance (-75%).
- **User Fold-In Validation**: We validated folding in new app users dynamically using a leave-one-out test on 200 users. Held-out wines ranked at a **0.92 mean percentile** (median 0.978), confirming that recommendations are personalized and not just a popularity echo.
- **Recipe Rating Accuracy**: Funk SVD achieved an RMSE of **0.6136** on Food.com test sets (compared to the baseline global mean RMSE of 1.12).
- **Lifecycle & Weights Simulation**: Simulated user journeys from 0 to 10 ratings to ensure the transition from cold-start to warm CF was smooth. Conducted grid searches to verify optimal component weights.
- **Test Coverage**: Maintained 530+ pytest backend tests and 63 Playwright frontend tests to ensure calculation and API correctness.

&nbsp;<br>

## Main Features

- **Hybrid Feeds with Score Explanations**: Custom recipe and wine suggestions with UI displays showing score breakdowns.
- **Dynamic Sorting**: Users can sort recipe results in the UI by individual score components (like expiry urgency or CF score).
- **Automated Wine Pairing**: Matches wines to recipes using a mix of 12-dimensional category similarity and empirical rules.
- **Pantry Scanner & Canonicalization**: Recognizes ingredients and expiry dates from fridge photos using GPT-4o or Gemini, automatically cleaning brand names.
- **Preference Learning**: Learns actual waste-aversion habits by tracking missing ingredients at cook time and adjusting the preference parameter ($\beta$).
- **Personalized Wine suggestions**: Tailored recommendations with style filters and rating feedback.
- **Recipe Skip Exclusions**: Hides skipped recipes for 7 days to avoid repetition.
- **Persistent Shopping List**: Syncs missing ingredients from recipes, tracking their source and allowing item check-offs.

