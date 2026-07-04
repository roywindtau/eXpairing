# Modules Description

This document details the architectural modules of exPairing across both the **Recipe Recommender System** and the **Drinks & Wine Recommender Module**, outlining technology stacks, core responsibilities, cross-module interactions, and source code locations.

## User Interface

- Technology: React 18, TypeScript, Vite, CSS Modules / Custom Properties, Axios, Lucide React icons.
- Responsibilities: Displays interactive SPA screens (Onboarding, Pantry, Ranked Recipe Feed, Recipe Detail with step-by-step instructions, Recipe Search/Browse, Drinks & Wine Feed, Profile Settings, and Persistent Shopping List). Collects user inputs, explicit star ratings (1-5 stars), and implicit cook/skip actions.
- Interactions: 
  - Makes REST calls via Axios client (`frontend/src/api/client.ts` and `frontend/src/api/wine.ts`) to backend endpoints (`/recipes/ranked`, `/pantry`, `/events`, `/wine/ranked`, `/wine/pair`, `/wine-events`, `/shopping`).
  - Receives JSON responses and dynamically renders score rings, match indicators, and component explainer bars.
- More info: Supports client-side sorting within the loaded 20 recommendations by individual score components (CF score, CB score, expiry urgency, pantry match) and provides style filters for wine.
- Source code: [`/frontend/`](../frontend/)

## API Gateway & Routers

- Technology: FastAPI, Python 3.9+, Pydantic, Uvicorn.
- Responsibilities: Exposes system REST endpoints, validates incoming JSON request payloads, manages database dependency sessions, and orchestrates call flows between data stores and machine learning serving engines.
- Interactions: 
  - Handles HTTP requests from the frontend client.
  - Passes request parameters to internal scoring services (`scoring.py`, `wine/scoring.py`), vision agents, and profile managers.
  - Returns structured, typed JSON responses to the frontend.
- More info: Sub-routers organized logically by domain: `pantry.py`, `recipes.py`, `users.py`, `vision.py`, `shopping.py`, and `wine.py`.
- Source code: [`/backend/routers/`](../backend/routers/)

## Database Schema & Data Models

- Technology: SQLite, SQLAlchemy ORM.
- Responsibilities: Manages persistent storage across 7 tables for both domains (`users`, `pantry_items`, `recipes`, `user_events`, `shopping_list_items`, `wines`, `wine_events`). Enforces foreign keys and cascades.
- Interactions: 
  - Queried by routers and scoring services for user state, active pantry contents, events, and shopping items.
- More info: Intentionally denormalizes `ingredients_csv` in `recipes` for fast $O(1)$ string parsing during candidate ranking (avoiding joins over 2M+ ingredient rows). Consolidates explicit ratings, implicit cook actions, and skip exclusions into a single `user_events` table using an event type discriminator.
- Source code: [`/backend/db/models.py`](../backend/db/models.py)

## Recipe Recommendation & Scoring Engine

- Technology: Python, NumPy, SciPy.
- Responsibilities: Implements the multi-component ranking formula: `final_score = γ·cf_score + δ·cb_score + α·expiry_urgency + β·match_ratio`. Min-max calibrates each score component across candidate pools and applies Maximal Marginal Relevance (MMR λ=0.7) diversity reranking on the top 60 candidates.
- Interactions: 
  - Fetches active pantry items, expiry dates, and diet tags from SQLite DB models.
  - Queries `serve_cf.py` (collaborative filtering) and `serve_cb.py` (content-based similarity).
  - Evaluates domain constraints from `expiry.py` and `ingredient_match.py`.
- More info: Ensures score calibration so wide-ranging domain signals do not drown out narrow CF distributions.
- Source code: [`/backend/services/scoring.py`](../backend/services/scoring.py)

## Recipe Collaborative Filtering Engine (CF)

- Technology: Scikit-surprise (Biased Funk SVD / SGD Matrix Factorization), SciPy sparse matrix operations (`csr_matrix`).
- Responsibilities: Predicts user taste preferences from user-item rating matrices. Features dual-strategy serving: personalized latent factor dot-products (`predicted(u,r) = μ + b_u + b_r + p_u·q_r^T`) for warm users (≥5 ratings) and item-item cosine similarity on mean-centered rating graphs for cold-start users (<5 ratings).
- Interactions: 
  - Evaluated at request time by `scoring.py`.
  - Loaded from pre-trained offline artifacts (`cf_model.pkl`, `item_sim_matrix.npz`).
- More info: Integrates synthetic ratings generated from cook events (`max(3.0, 4.0 - n_missing*0.3)`) into matrix factorization training data when explicit ratings are absent.
- Source code: [`/backend/ml/serve_cf.py`](../backend/ml/serve_cf.py) & [`/backend/ml/cold_start.py`](../backend/ml/cold_start.py)

## Recipe Content-Based Filtering Engine (CB)

- Technology: Scikit-learn (`TfidfVectorizer`, `cosine_similarity`).
- Responsibilities: Encodes recipe ingredient lists into 20,000-dimensional TF-IDF feature vectors (unigrams + bigrams). Maintains user taste profile vectors constructed from current pantry items (cold start) or rating-weighted historical recipe vectors (warm start).
- Interactions: 
  - Provides candidate-to-profile cosine similarity scores to `scoring.py`.
  - Reads pre-computed recipe vectors from `cb_matrix.npz`.
- More info: Captures cuisine affinities (e.g., miso + soy sauce matching Japanese dishes) without requiring explicit cuisine tag metadata.
- Source code: [`/backend/ml/serve_cb.py`](../backend/ml/serve_cb.py) & [`/backend/ml/train_cb.py`](../backend/ml/train_cb.py)

## Drinks & Wine Recommender Engine

- Technology: Implicit (Confidence-Weighted ALS Matrix Factorization), Structured Vector Cosine Matching.
- Responsibilities: Powers the personalized "Suggest me a wine" feature by blending collaborative filtering (confidence-weighted ALS trained on 21M X-Wines ratings) and structured content-based attribute matching (grape multi-hot, parent region rollup, body, acidity, abv) with a Bayesian popularity prior.
- Interactions: 
  - Exposes endpoints `/wine/ranked` and `/wine-events`.
  - Evaluates cold (0 ratings → popularity), warming (1-4 ratings → CB + popularity), and warm (≥5 ratings → 0.45·CF + 0.45·CB + 0.10·popularity min-max calibrated) user states.
- More info: Incorporates sommelier palate-first priors where structural attributes (body + acidity ~74% weight) guide content similarity. Folds in active app users at runtime via online ALS solver updates.
- Source code: [`/backend/routers/wine.py`](../backend/routers/wine.py), [`/backend/services/wine/scoring.py`](../backend/services/wine/scoring.py), & [`/backend/ml/wine/serving/`](../backend/ml/wine/serving/)

## Recipe-Wine Pairing Module

- Technology: Python, NumPy, SciPy sparse vectors.
- Responsibilities: Powers the automated recipe-wine pairing engine (`POST /wine/pair`). Maps recipe ingredients onto a 12-dimensional food category vector (`recipe_categories.py`) and ranks wines by blending category vector cosine similarity (`ALPHA_COSINE=0.6`) with empirical sommelier pairing rules (`BETA_RULES=0.4`) extracted from labeled pairing matrices (`pairing_rules.json`).
- Interactions: 
  - Invoked via router endpoint `POST /wine/pair`.
  - Reads pre-computed wine pairing matrices (`models/wine_pair_matrix.npz`) and empirical rules.
- More info: Applies MMR reranking to the top candidate pool to ensure diverse bottle recommendations for any recipe.
- Source code: [`/backend/ml/wine/serving/serve_pairing.py`](../backend/ml/wine/serving/serve_pairing.py) & [`/backend/routers/wine.py`](../backend/routers/wine.py)

## User Profile Manager & Preference Learning (Beta Updater)

- Technology: Python, SQLAlchemy, Exponential Moving Average (EMA) math.
- Responsibilities: Manages persistent user profile records (diet tags, stated beta preferences) and executes daily batch preference learning (`beta_updater.py`). Dynamically tracks missing ingredient counts (`n_missing`) during cooking actions and updates individual waste-aversion preferences (`β`) via EMA (`new_β = 0.85·current_β + 0.15·revealed_β`).
- Interactions: 
  - Consulted by scoring services on every request for per-user weighting parameters.
  - Triggers profile warnings when stated beta and revealed beta diverge by > 10%.
- Source code: [`/backend/services/beta_updater.py`](../backend/services/beta_updater.py) & [`/backend/db/models.py`](../backend/db/models.py)

## Expiry Urgency & Ingredient Matcher

- Technology: Python, RapidFuzz.
- Responsibilities: Computes exponential expiry urgency scores (`exp(-k · days)`) normalized by pantry size to prioritize waste minimization. Performs fuzzy ingredient string matching (threshold 75) to calculate ingredient availability ratios and generate lists of missing ingredients.
- Interactions: 
  - Called directly by `scoring.py` during candidate ranking.
- Source code: [`/backend/services/expiry.py`](../backend/services/expiry.py) & [`/backend/services/ingredient_match.py`](../backend/services/ingredient_match.py)

## Vision Agent & Ingredient Canonicalizer

- Technology: OpenAI GPT-4o API (`OPENAI_API_KEY`), Google Gemini 2.5 Flash API (`GEMINI_API_KEY`), RapidFuzz, OpenFoodFacts API.
- Responsibilities: Processes user fridge photos via multi-modal AI vision APIs to extract item labels, quantities, and printed expiration dates (`YYYY-MM-DD`). Cleans store packaging noise and maps raw brand names to canonical Food.com vocabulary tokens.
- Interactions: 
  - Invoked via endpoint `/vision/scan` (or `/vision/mock` for API-key-free development). Returns canonicalized items for insertion into SQLite pantry tables.
- More info: Two-step canonicalization pipeline strips brand names ("Tnuva", "Heinz") and noise adjectives, then fuzzy-matches generic items against the 20,000-token recipe ingredient vocabulary (`FUZZY_THRESHOLD=70`).
- Source code: [`/backend/services/vision_agent.py`](../backend/services/vision_agent.py) & [`/backend/canonicalizer/`](../backend/canonicalizer/)

## Persistent Shopping List Manager

- Technology: Python, SQLAlchemy, FastAPI.
- Responsibilities: Manages a persistent buy-list for users. Allows adding missing ingredients directly from recipe detail cards, handles deduplication, item check-offs, clear-purchased actions, and maintains source recipe attribution.
- Interactions: 
  - Communicates via `/shopping/{user_id}` REST endpoints.
- Source code: [`/backend/routers/shopping.py`](../backend/routers/shopping.py)
