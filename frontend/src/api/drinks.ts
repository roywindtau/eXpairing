// drinks.ts
// Typed wrappers around the backend drink endpoints (Step 8).

import api from './client'

// ── types matching backend Pydantic models ─────────────────────────────────

export type DrinkKind = 'wine'
export type KindFilter = DrinkKind | 'all'

// Strategy labels returned by backend/ml/serve_drink_cf.cf_strategy_name().
// Keep this union in sync with that function.
export type CfStrategy =
  | 'popularity_cold_start'  // user has no explicit drink ratings yet
  | 'wine_item_sim'          // wines use item-sim from history
  | 'none'

export interface DrinkScoreOut {
  drink_id:     number
  drink_name:   string
  kind:         DrinkKind
  final_score:  number
  cb_score:     number
  cf_score:     number
  expert_boost: number
  prior_score:  number
  cf_strategy:  CfStrategy
  avg_rating:   number | null
  n_ratings:    number
  abv:          number | null
  producer:     string | null
  style:         string | null
  grapes_csv:    string | null
  harmonize_csv: string | null
}

export interface DrinkSearchHit {
  id:            number
  name:          string
  kind:          DrinkKind
  style:         string | null
  grapes_csv:    string | null
  harmonize_csv: string | null
  producer:      string | null
  abv:           number | null
  avg_rating:    number | null
  n_ratings:     number
}

export interface DrinkDetail {
  id: number
  name: string
  kind: DrinkKind
  producer: string | null
  country: string | null
  abv: number | null
  avg_rating: number | null
  n_ratings: number
  style: string | null
  // wine
  grapes_csv: string | null
  region: string | null
  body: string | null
  acidity: string | null
  harmonize_csv: string | null
}

// ── API helpers ─────────────────────────────────────────────────────────────

/** Path B — "For You" rankings (uses user history). */
export const getRankedDrinks = (
  userId: number,
  kind: KindFilter = 'all',
  topN = 24,
) =>
  api.get<DrinkScoreOut[]>('/drinks/ranked', {
    params: { user_id: userId, kind, top_n: topN },
  }).then(r => r.data)

/** Path A — pair drinks with a specific recipe. */
export const getDrinkPairings = (
  recipeId: number,
  userId: number,
  kind: KindFilter = 'all',
  topN = 6,
) =>
  api.get<DrinkScoreOut[]>(`/drinks/pairings/${recipeId}`, {
    params: { user_id: userId, kind, top_n: topN },
  }).then(r => r.data)

export const searchDrinks = (q: string, kind: KindFilter = 'all', limit = 40) =>
  api.get<DrinkSearchHit[]>('/drinks/search', {
    params: { q, kind, limit },
  }).then(r => r.data)

export const getDrinkDetail = (drinkId: number) =>
  api.get<DrinkDetail>(`/drinks/${drinkId}`).then(r => r.data)

export const rateDrink = (userId: number, drinkId: number, rating: number) =>
  api.post<{ status: string; event_id: number }>('/drink-events', {
    user_id: userId,
    drink_id: drinkId,
    event_type: 'rate',
    rating,
  }).then(r => r.data)
