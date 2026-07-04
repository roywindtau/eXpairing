# Project Summary

## Product Overview & Value Proposition (Product POV)

### The Real-World Problem Solved
Food waste is a major financial, logistical, and environmental challenge in modern households. Pantries and fridges are full but invisible — consumers frequently purchase ingredients, put them away, and forget they exist until they have already expired. Even when users notice an item approaching its expiration date, they struggle to spontaneously recall a recipe that utilizes it, resulting in preventable food waste and unnecessary grocery expenditures.

Traditional recommender systems and recipe platforms ask: *"What sounds good to eat?"* or *"What do you feel like cooking?"*  
**exPairing completely re-engineers this paradigm.** Built like a startup product for the Tel Aviv University Recommender Systems Workshop, exPairing operates as an intelligent culinary assistant that asks:
> *"Given what is expiring in your fridge right now and how you actually cook, what is the highest-quality meal you will thoroughly enjoy?"*

### Key Product Value Drivers
1. **Dual-Signal Balancing (Preference vs. Feasibility)**: Combines machine learning preference predictions (what you enjoy) with physical household constraints (what is expiring and available in your fridge right now).
2. **Seamless Multi-Modal Inventory Ingestion**: Multi-modal vision scanning (GPT-4o / Gemini 2.5 Flash) and debounced ingredient autocomplete eliminate the friction of manual pantry logging. An automated canonicalization engine strips store packaging noise ("Tnuva 3% Milk") to map items onto standardized recipe tokens ("milk").
3. **Behavioral Adaptation & Revealed Preference Learning**: Recognizes that users often exhibit aspirational bias when setting waste aversion preferences ($\beta$). exPairing tracks revealed cooking behavior (`n_missing`) and smoothly drifts $\beta$ via Exponential Moving Average (EMA) batch updates, displaying a profile alert when stated and revealed preferences diverge by $>10\%$.
4. **End-to-End Culinary & Dining Experience**: Extends beyond recipe recommendation into a full household utility. Includes interactive step-by-step cooking instructions, persistent shopping list management with one-click check-offs, 7-day skip exclusion memory, and a dedicated **Drinks & Wine Recommender** that pairs wines directly with recipes or suggests personalized bottles matching user palates.

&nbsp;<br>

## Datasets Used

- **Food.com Recipes and User Interactions** — Primary dataset for the recipe recommender, [Kaggle Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). Contains 231,637 recipes, 1,132,367 user reviews/ratings (1–5 explicit stars), ingredient tokens, tags, cooking minutes, and step-by-step instructions.
- **X-Wines Dataset (Full)** — Primary dataset for the drinks/wine recommender, [GitHub Repository](https://github.com/rogerioxavier/X-Wines). Contains 100,646 wines and 21 million user ratings, along with structured metadata (grapes, 2,160 appellations, body, acidity, alcohol content).
- **Labeled Wine-Food Pairing Dataset** — Training input for the recipe-wine pairing engine (`data/pairing/wine_food_pairings.csv`, ~35K rows committed to the repo). Each row labels a (wine category × food category) combination with a 1-5 pairing-quality score. Signal analysis (`data/pairing/check_ingredient_signal.py`) showed the labels are category-level rule-generated, so we extract the underlying rule table directly — per-cell mean quality with injected contrast rows dropped (`data/pairing/extract_pairing_rules.py` → `models/pairing_rules.json`) — rather than fit a model to ingredient-level noise.

Additional data-related information:
- **Canonicalization & Entity Resolution**: Engineered a rule-based canonicalizer (`backend/canonicalizer/ingredient_map.py`) and integrated OpenFoodFacts API to resolve real-world store item names into Food.com canonical vocabulary tokens (e.g. mapping brand variants like "Tnuva 3% Milk" to canonical "milk", and synonyms like "aubergine" to "eggplant").
- **Region Rollup (Wine Domain)**: Developed `data/wine/region_rollup.py` to collapse 2,160 raw wine appellations down to 107 parent regions (e.g. Pauillac → Bordeaux, Meursault → Burgundy), enabling meaningful content-based region overlap.
- **Multi-Modal AI Extraction**: Integrated GPT-4o Vision and Gemini 2.5 Flash to parse physical fridge photographs, extract item lists with expiration dates, and insert them directly into our database schema.

&nbsp;<br>

## Technologies and Frameworks

### Frontend
- **React 18 & TypeScript** — component-based single-page application with static type safety.
- **Vite** — frontend build tool and hot-reloading development server.
- **CSS Modules & Custom Properties** — custom styling, score ring graphics, and responsive layouts.
- **Axios** — HTTP client for asynchronous REST API communication (`client.ts` and `wine.ts`).
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
- **Deterministic Mock Scanner** — dev/demo fallback allowing complete offline execution without requiring paid API tokens.

&nbsp;<br>

## Main Algorithms & Core Rationale

A summary of the primary algorithms developed across both Recipe and Drinks domains, highlighting what each algorithm does and why it was ideal for our system:

### 1. Recipe Collaborative Filtering — Biased Funk SVD (Matrix Factorization)
- **What it does**: Predicts personalized user ratings for recipes ($\ge 5$ ratings) by learning 50 latent preference factors per user and recipe, while explicitly accounting for user rating scale bias ($b_u$) and recipe popularity bias ($b_r$) via $\mu + b_u + b_r + p_u \cdot q_r^T$.
- **Why it was ideal**: Designed specifically for highly sparse explicit rating matrices (~99.998% empty). Training on observed ratings via Stochastic Gradient Descent (SGD) avoided the heavy noise and computational burden of dense matrix imputation.

### 2. Recipe Cold-Start CF — Preference-Seeded Item Similarity
- **What it does**: Evaluates cold-start users ($< 5$ ratings) by mapping user dietary tags and pantry items onto a pre-computed sparse item-item cosine similarity matrix built from co-rating patterns across 51k+ recipes.
- **Why it was ideal**: Solves user cold-start without relying on unpopulated user vectors or generic, unpersonalized popularity lists. Using pantry items as preference anchors allows content signals to bootstrap recommendations directly into the behavioral item graph space.

### 3. Recipe Content-Based Engine — TF-IDF Profiling
- **What it does**: Represents recipe ingredient lists as 20,000-dimensional TF-IDF vectors and computes cosine similarity against user taste profile vectors (built from pantry contents for cold users, or rating-weighted historical recipe vectors for warm users).
- **Why it was ideal**: Automatically highlights distinctive ingredients (saffron) while discounting common staples (salt). Exact lexical matching over canonical tokens is superior to dense text embeddings in an inventory system, as having butter in the fridge should prioritize actual butter recipes rather than semantic substitutes like margarine.

### 4. Drinks & Wine Collaborative Filtering — Confidence-Weighted ALS
- **What it does**: Ranks wines for personalized feeds ("Suggest me a wine") using confidence-weighted Alternating Least Squares ($C = 1 + 5\cdot\text{rating}$, factors=64, reg=0.05) trained on 21M X-Wines ratings. Solves online user factor updates at runtime for active app users.
- **Why it was ideal**: Wine data consists of explicit ratings without implicit interaction logs. ALS treats unobserved items as confidence-weighted weak negatives, making it natively suited for ranking un-rated items (achieving an NDCG@10 of 0.0291, 4× popularity floor).

### 5. Drinks & Wine Content-Based Model — Sommelier Structured Vectors
- **What it does**: Constructs content vectors across structured wine attributes (grape multi-hot, parent region rollup, body, acidity, abv) weighted by sommelier palate-first priors where structural attributes (body + acidity $\sim 74\%$ weight) guide cosine similarity.
- **Why it was ideal**: X-Wines contains zero free-text reviews, making NLP embeddings inapplicable. A structured vector with sommelier-weighted structural blocks provides highly interpretable attribute matching.

### 6. Automated Recipe-Wine Pairing Engine — 12-Dim Category Vectors & Empirical Rules
- **What it does**: Automatically pairs wines with recipes (`POST /wine/pair`) by projecting recipe ingredients onto a 12-dimensional food category vector space (`recipe_categories.py`) and blending category cosine similarity (`ALPHA_COSINE=0.6`) with empirical sommelier pairing rules (`BETA_RULES=0.4`) extracted from labeled pairing matrices (`pairing_rules.json`).
- **Why it was ideal**: Bridges the cross-domain gap between food ingredients and wine styles by aligning both items in a shared 12-category culinary space.

### 7. Domain Constraints & Diversity Reranking
- **Expiry Urgency & Match Penalization**: Adjusts rankings using exponential expiry decay ($\exp(-k \cdot \text{days})$) and per-user ingredient availability match ratios ($\beta$), prioritizing waste reduction without overriding base preferences.
- **Maximal Marginal Relevance (MMR λ=0.7)**: Greedily reranks top candidate pools using pairwise ingredient/wine cosine similarity to balance recommendation relevance against feed diversity.

&nbsp;<br>

## Database Schema & Data Layer Structure

The persistent storage layer is implemented in SQLite via SQLAlchemy ORM (`backend/db/models.py`), organized into 7 tables across both domains to balance normalization and real-time query performance:

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
6. **`wines`**: Flat catalog table for the drinks module (100,646 rows from X-Wines). Stores `id`, `winename`, `wine_type`/`style`, `vintage`, `abv`, `acidity`, `body`, `country`, `region`, `avg_rating`, and `n_ratings`.
7. **`wine_events`**: Event logging table for drinks. Stores `user_id`, `wine_id`, `event_type` (`"rate"`), and star `rating` (1–5).

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
3. **Multi-Model Scoring (Stage 2)**: For each candidate, evaluates four distinct signals:
   - *Collaborative Filtering Score*: Automatically selects item-based CF (cold users 0 ratings) or biased Funk SVD matrix factorization (warm users ≥5 ratings). Between 1 and 4 ratings, blends both via `alpha = n_ratings / 5` to ensure smooth transitions.
   - *Content-Based Score*: Cosine similarity between candidate ingredient TF-IDF vector and user profile vector.
   - *Expiry Urgency Score*: Exponential decay calculation based on pantry items expiring soonest.
   - *Ingredient Match Ratio*: Fuzzy overlap between recipe requirements and available pantry items.
4. **Min-Max Calibration & Weighted Blending**: Each score component is min-max normalized across the candidate pool before blending. This is critical because raw CF scores cluster in a narrow range (0.30–0.38) while expiry urgency spans broadly (0.02–0.95); calibration ensures assigned weights (35% CF, 35% expiry, 20% match, 10% CB) maintain true proportions.
5. **MMR Reranking & Feedback Loop**: Top 60 candidates pass through MMR diversity reranking (λ=0.7) using ingredient Jaccard similarity. Top 20 recipes returned to UI with complete score breakdowns. User actions (`cook`, `rate`, `skip`) are saved to SQLite, feeding synthetic rating generators (`max(3.0, 4.0 - n_missing*0.3)`) and daily preference updates (`β`).

**System Flow (Request Lifecycle for `GET /wine/ranked` & `POST /wine/pair`):**
1. **Personalized Wine Feed (`GET /wine/ranked`)**: Applies hard style filters and checks user rating counts. *Cold start* (0 ratings) → Bayesian popularity prior. *Warming* (1-4 ratings) → Content-based taste profile + popularity. *Warm* (≥5 ratings) → `0.45·ALS_CF + 0.45·CB + 0.10·popularity` min-max calibrated. Active app users are folded in dynamically by solving online ALS user updates (`C = 1 + 5·rating`) against frozen item factors.
2. **Automated Recipe-Wine Pairing (`POST /wine/pair`)**: Accepts a `recipe_id`, converts its ingredients to a 12-dim food category vector, and ranks wines by combining category cosine similarity (`ALPHA_COSINE=0.6`) with empirical sommelier pairing rules (`BETA_RULES=0.4`), applying MMR reranking for bottle diversity.

## Development Environment
- **Cursor & VS Code** - used for React frontend UI development, FastAPI backend service orchestration, and ML training scripts.
- **Pytest & Playwright** - used for backend unit testing (530+ tests) and end-to-end browser behavioral verification (63 tests).

&nbsp;<br>

## Development Evolution

- **Milestone 1: Foundations & Rule-Based MVP**
  Created SQLite database schema, FastAPI scaffolding, and React single-page interface. Implemented baseline candidate fetching and rule-based ingredient overlap matching.
- **Milestone 2: Content-Based Engine & Domain Constraints**
  Fitted TfidfVectorizer over 231k Food.com recipes. Implemented exponential expiry urgency decay based on pantry dates and introduced per-user `β` parameter to penalize missing ingredients.
- **Milestone 3: Collaborative Filtering & Cold Start Solution**
  Trained Biased Funk SVD matrix factorization on 1.1M explicit ratings using scikit-surprise, achieving RMSE 0.6136. Built sparse item-item similarity matrix across 51k recipes and engineered a preference-seeded cold start mechanism for new users.
- **Milestone 4: Score Calibration & Diversity Reranking**
  Discovered wide-ranging expiry scores drowned out narrow CF distributions. Introduced per-candidate min-max score calibration before blending. Integrated Maximal Marginal Relevance (MMR) reranking to ensure feed variety.
- **Milestone 5: Implicit Feedback Loops, Dual-Domain Wine Engine & Recipe Pairing**
  Converted cook events into synthetic rating augmentations (`max(3.0, 4.0 - n_missing*0.3)`). Built `beta_updater.py` with EMA learning to adjust waste-aversion preferences automatically. Integrated X-Wines dataset, conducted ALS vs SVD algorithm bake-offs, built region rollup algorithms, deployed personalized wine recommendations, engineered the automated 12-dim recipe-wine pairing engine (`serve_pairing.py`), and implemented multi-modal AI vision scanning with dual API token support (`OPENAI_API_KEY` / `GEMINI_API_KEY`).

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

## Main Features

- **Multi-Domain Hybrid Ranking Feeds**: Personalizes recipe and wine feeds with transparent score breakdown rings and explainer bars.
- **Automated Recipe-Wine Pairing**: Pair wines directly with any recipe via `POST /wine/pair`, using a hybrid of 12-dimensional category vector cosine similarity and empirical sommelier rule matrices.
- **Dynamic Feed Re-Sorting**: Client-side controls allow users to re-sort loaded recipe recommendations dynamically by individual components (e.g. sort strictly by expiry urgency or CF score).
- **Multi-Modal AI Vision Scanner with Dual Token Support & Canonicalization**: Users take photos of their fridge; GPT-4o (`OPENAI_API_KEY`) or Gemini 2.5 Flash (`GEMINI_API_KEY`) extracts ingredients and expiry dates. Raw packaging text is stripped of brand names ("Tnuva", "Danone", "Heinz") and fuzzy-matched onto the 20k Food.com canonical vocabulary.
- **Revealed Preference Learning (`β` Updater)**: Tracks missing ingredient counts (`n_missing`) when users cook. An EMA background process updates the user's `β` weight, adapting recommendations to match actual cooking habits over time.
- **Personalized Drinks & Wine Module**: Provides tailored wine recommendations ("Suggest me a wine") with style selection chips, food pairing groupings, and star rating feedback.
- **Skip Exclusion Memory**: Suppresses skipped recipes from appearing in the user feed for 7 days to prevent recommendation repetition.
- **Persistent Shopping List Integration**: Allows users to add missing recipe ingredients directly to a persistent buy-list, complete with check-off mechanics and source recipe attribution.

## Open Issues, Limitations, and Future Work

- **Offline CF Retraining Schedule**: Matrix factorization weights are currently trained offline on static Food.com and X-Wines ratings. While in-app ratings and cook events are captured in SQLite, updating latent vectors requires triggering an offline retraining script. A scheduled background retraining pipeline is planned.
- **Sub-Millisecond Candidate Retrieval**: Candidate generation currently uses database indexing and popularity caps in SQLite. Migrating to vector search engines (such as FAISS or Qdrant) would allow sub-millisecond similarity queries across millions of items.
- **Noisy Grape Labels (X-Wines)**: Grape variety tags in X-Wines are noisy (e.g. a Cabernet blend tagged "Pinot Noir"), which affects the grape block of the wine content vector and any grape-based UI text. The sommelier weighting deliberately keeps the grape block small (~5%), limiting the impact.
- **Untested CF Lever — Positive-Rating Cut**: Dropping ratings <4 from the ALS confidence matrix (so disliked wines stop acting as weak positives) is the one identified pure-CF lever left untested; it could push NDCG@10 past the current 0.0291 ceiling.

&nbsp;<br>

## Additional Comments

Building exPairing provided several practical engineering insights regarding recommender systems:
1. **Calibration is Essential in Hybrid Systems**: Simply summing outputs from different models creates severe distortion. Collaborative filtering scores often cluster in a narrow range (e.g. 0.30–0.38), whereas expiry urgency spans from 0.02 to 0.95. Without per-candidate min-max calibration, expiry urgency dominated rankings regardless of assigned weights.
2. **Matching Algorithms to Objectives (NDCG vs RMSE)**: Evaluated Funk SVD vs ALS on wine data. SVD optimized RMSE (rating value accuracy) but failed at ranking unobserved items. ALS optimized ranking via confidence weighting, achieving 4× popularity NDCG@10. Recommender systems that rank items must optimize ranking metrics rather than prediction accuracy alone.
3. **TF-IDF vs. Embeddings in Ingredient Recommenders**: Pre-trained dense embeddings (Word2Vec, BERT) capture semantic similarity (e.g. `butter ≈ margarine`). However, in a fridge recommender, semantic substitution is undesirable: having *butter* in the fridge means the system must prioritize recipes requiring *butter*, not boost recipes calling for margarine. TF-IDF exact lexical matching over canonical vocabulary is strictly superior for inventory-matching tasks.
