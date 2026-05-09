# Intro to Recommender Systems

*Dr. Rubi Boim*

---

## TLDR

**Recommending by:**
1. Generate a list of candidate items
2. For all items, predict the probability of the event
3. Select the top-k with the highest score (or apply diversity / …)

**Recommender system types:**
- Content Based
- Collaborative Filtering
- Hybrid

---

## Agenda

1. Intro and Intuition
2. Content Based
3. Collaborative Filtering
4. Common Challenges

---

## Part 1: Intro and Intuition

### How Can You Build a Recommender System?

#### Attempt 1 — Manual editorial curation
An editor manually selects the item order.

**Problem:** All users get the same recommendations.

---

#### Attempt 2 — Recommend weekly trending
Recommend the most popular items this week.

**Problem:** Still all users get the same recommendations (though it may change each day).

**How to calculate it:**
1. Count the number of views for all movies:
   - Top Gun: 8,232
   - The Matrix: 103,294
   - Inception: 32,124
   - …
2. Order the list and select top 4.

> Congratulations! This is the first "model" you built 🙂

**Offline / Training / Building** can take minutes / hours / days.  
**Online / Serving / Inference** runs in milliseconds.

---

#### Attempt 3 — Recommend weekly trending per country
Still not really personalized (how many people are in India?).

---

#### Attempt 4 — Personalized prediction
For each user, predict which items they will like.

Finally… **but how do you do it???**

---

### Recommender System 101

- **Generate a list of candidate items**
  e.g., all movies licensed in Israel which the user has not seen
- **For all items, predict the probability of the event**
  e.g., Top Gun 2: 0.84, The Matrix: 0.62, American Pie: 0.94
- **Select the top-k with the highest score** (or apply diversity / …)
  e.g., American Pie, Top Gun 2

---

### Recommender System Types

| Type | Based on | Intuition |
|------|----------|-----------|
| **Content Based** | Semantic / static properties | "Because you watch comedy movies with Adam Sandler, here are more comedies you might like" |
| **Collaborative Filtering** | User behavior | "Other users who watched *Grown Ups* also watched…" |
| **Hybrid** | Mix of both | Predict new genres a user might like |

**Key insight about Collaborative Filtering:**
> CF works without any prior semantic knowledge. You can blend any type of item together in the same algorithm (movies, food, books, hotels…).

---

### A Short History Lesson: "The Netflix Prize"

In 2006, Netflix offered **$1M** to any team that could improve their recommender system by 10% (RMSE error).

- **First "big" public dataset:** 0.5M users, 17K movies, 100M ratings (1–5)
- **Huge leap forward** for recommender systems and ML competitions → Kaggle started in 2010

It was proven that **CF is better than CB** — *this is counter-intuitive*. To predict movie ratings, you do not need to know *any* semantic information (actor, genre, year…).

> In practice, you would use a hybrid approach.

---

## Part 2: Content Based

### How Content-Based Filtering Works

1. **Build a profile for each item** using its features (attributes): genres, categories, actors, release date, price, …
2. **Build a profile for each user** by extracting features from the items they previously interacted with (explicit / implicit)
3. **Recommend items with similar profiles to the user**

---

### Feature Extraction (by Data Type)

| Data Type | Examples | Processing |
|-----------|----------|------------|
| Structured metadata | genres, categories, actors, release date, price | Already structured — just filter as needed |
| Free text | description, plot | Extract features via NLP pipeline |
| Visual | images, videos | Convert to free text description, then extract |

#### NLP Pipeline for Free Text

**Example text:**
> "Top Gun: Maverick is a 2022 American action drama film directed by Joseph Kosinski and written by Ehren Kruger, Eric Warren Singer, and Christopher McQuarrie. A sequel to the 1986 film Top Gun, Tom Cruise reprises his starring role as the naval aviator Maverick."

**Step 1 — Tokenization:**
`[Top, Gun, Maverick, is, a, 2022, American, action, drama, film, directed, by, Joseph, Kosinski, and, written, by, Ehren, Kruger, Eric, Warren, Singer, and, Christopher, McQuarrie, A, sequel, to, the, 1986, film, Top, Gun, Tom, Cruise, reprises, his, starring, role, as, the, naval, aviator, Maverick]`

**Step 2 — Removing Stop Words:**
`[Top, Gun, Maverick, 2022, American, action, drama, film, directed, Joseph, Kosinski, written, Ehren, Kruger, Eric, Warren, Singer, Christopher, McQuarrie, sequel, 1986, film, Top, Gun, Tom, Cruise, reprises, starring, role, naval, aviator, Maverick]`

**Step 3 — Stemming:**
`[top, gun, maverick, 2022, american, action, drama, film, direct, joseph, kosinski, writ, ehren, kruger, eric, warren, singer, christopher, mcquarri, sequel, 1986, film, top, gun, tom, cruise, reprais, starr, role, naval, aviator, maverick]`

---

### Vectorization

Converting tokens into numerical representations so they can be processed by ML.

#### BoW / TF-IDF (simple and effective)

- Create a dictionary of all words (the "space") — *without word order*
- **Bag of Words (BoW):** counts the number of appearances of each word in the document; 1 if present in structured data
- **TF-IDF:**
  - *Term Frequency* — weight by frequency in the document
  - *Inverse Document Frequency* — measures how common or rare a term is across all documents in the corpus

**TF-IDF Notes:**
- Vector size = dictionary size → **high dimension**
- Most values are zero → **sparse data**
- Requires exact text match: `"tel aviv" ≠ "tel-aviv"`, `"car" ≠ "automobile"`

#### Embeddings (complex, superior results)

Dense vector representations where each feature is mapped into a continuous vector space.

- Generated using neural network models
- Capture semantic meaning by placing similar words/documents closer together in embedding space
- Popular models: Word2Vec, Sentence-BERT, embed-english-v3.0 (Cohere), text-embedding-ada-002 (OpenAI)

**Embeddings Notes:**
- Fixed, relatively small size: 300–2,000 in most modern models
- Most values are non-zero → **dense**
- Captures semantics: `"tel aviv" ≈ "tel-aviv"`, `"car" ≈ "automobile"`

---

### TF-IDF vs Embeddings

| Aspect | TF-IDF | Embeddings |
|--------|--------|------------|
| Representation | Sparse, high-dimensional | Dense, low-dimensional |
| Interpretability | Highly interpretable (words are explicit) | Less interpretable (dimensions are abstract) |
| Semantic Capture | Limited | Rich |
| Computational Cost | Relatively low | Higher (especially during training) |
| Best For | Smaller datasets, interpretability is key | Larger datasets, capturing nuance is critical |

---

### Cosine Similarity

Used to compare two vectors — popular choice for TF-IDF and Embeddings.

- Calculates the cosine of the angle between two vectors
- If vectors point in the same direction (angle = 0°), cosine similarity = 1 (maximum similarity)

---

## Part 3: Collaborative Filtering

### Overview

Recommend based on the behavior and preferences of other users.

**Memory-based CF:**
- User-based CF
- Item-based CF

**Model-based CF:**
- Matrix Factorization

---

### The User-Item Matrix

A matrix where rows represent users, columns represent items, and cells represent events (e.g., VIEW, DOWNLOAD, RATING).

**Example:** Jordan viewed Moana 2 and Grown Ups → both cells marked in Jordan's row.

The matrix is used to find users or items with similar interaction patterns.

---

### User-Based CF

**Intuition:** The preferences of similar users (neighbors) can predict preferences for an item.

**Steps:**
1. **Similarity calculation** — compute similarity with all other users (Cosine / Pearson / Jaccard …)
2. **Neighborhood formation** — identify a set of similar users (neighbors)
3. **Prediction** — take a weighted average of the neighbors' preferences

---

### Item-Based CF

**Intuition:** If two items are "similar," a user who liked one will likely like the other.

**Steps:**
1. **Similarity calculation** — compute similarities with all other items (Cosine / Pearson / Jaccard …)
2. **Neighborhood formation** — identify a set of similar items (neighbors)
3. **Prediction** — take a weighted average of the preferences by neighbors

**Example:** Recommend Interstellar because you watched Moana 2 — even if one is a kids movie and the other is SciFi.
> "Numbers don't lie 🙂 — this is why CF is better than CB."

> **Note:** Similarity means something different in CB vs CF.

---

### User-Based vs Item-Based CF

**TLDR: Item-based is preferred most of the time.**

| Aspect | User-Based CF | Item-Based CF |
|--------|---------------|---------------|
| Similarity Calculation | Between users | Between items |
| Scalability | More computationally intensive with many users | Typically more scalable (fewer items than users) |
| Stability | Volatile as user preferences change | More stable — item properties change slowly |
| Data Sparsity | Can struggle with users who have few interactions | Often more robust — items accumulate more interactions |
| Recommendation Focus | Based on similar users' behavior | Based on items similar to what the user engaged with |

---

### Matrix Factorization

One of the most popular techniques for CF.

**What it does:**
- Discovers **latent factors** that explain user-item interactions
  - latent factors ≈ hidden dimensions ≈ "embeddings"
- Decomposes the large, sparse user-item matrix **R** into two (or three) lower-dimensional matrices:
  - **U** — User features matrix
  - **V** — Item features matrix
  - Such that **R ≈ U × Vᵀ**

**Prediction:**
The predicted rating/preference for user *u* and item *i* is given by the **dot product** of their latent vectors: `r̂(u,i) = U_u · V_i`

A popular extension (from the Netflix Prize) adds **bias values** to account for user and item-level tendencies.

#### Matrix Factorization Techniques

| Technique | Description |
|-----------|-------------|
| **SVD** | Full decomposition using orthogonal matrices and singular values |
| **Funk SVD** | Optimization approach using gradient descent |
| **PMF** | Probabilistic approach to factorization |
| **NMF** | Ensures all elements are non-negative for interpretability |
| **ALS** | Alternates solving for user and item matrices using least squares |

---

## Part 4: Common Challenges

### Implicit vs Explicit Data

**Explicit data:** asking a user directly ("what are 3 movies you like?")
- Subject to social desirability bias — is it "cool" to say Moana 2 or Taylor Swift?

**Implicit data:** behavioral signals — clicks, views (50%+), searches, shares, copy link, likes, …
- Captures your **true** preference → always works better 🙂

---

### Data Sparsity

The user-item matrix is extremely sparse.

**Netflix Prize example:**
- 0.5M users × 17.7K movies = ~9 billion possible values
- Only 100M ratings provided → **~99% of the matrix is unknown**

Recommender system datasets are:
- **Big Data** (endless implicit events)
- **But SPARSE**

> Traditional ML algorithms do NOT work on sparse data — this is why Recommender Systems are unique.

---

### Multi-Event Data

Can a single matrix represent multiple event types (VIEW, DOWNLOAD, RATING)?

Yes — techniques exist to "blend" or "merge" events, for example:
- +1 for "view"
- +2 for "share"
- -1 for "dislike"

**KISS — Keep It Simple, Stupid!**  
→ Use a single event, at least the first time.

---

### Cold Start

**Problem:** What to do when a *new* user enters the system?

For CF, options are limited by definition:
- Show average/popular items
- Use available semantics (location, gender, age…)
- Ask for explicit data
- Build a real-time system that kicks in after the first event

**Can you have a Cold Start problem for new items?**  
**YES — it is an even bigger problem!**
- You need to retrain the ML model
- Possible outcome: new items never get recommended to any user

---

### Diversity

Without diversity mechanisms, a recommender can fall into a "filter bubble" — showing the same types of content repeatedly.

**How to implement diversity:**  
Easier with metadata / semantics / CB — so if you have it, use it.

---

*Workshop slides by Dr. Rubi Boim*
