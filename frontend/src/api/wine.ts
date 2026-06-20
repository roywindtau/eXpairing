// wine.ts
// Typed wrappers around the backend wine endpoints.

import api from './client'

// ── types matching backend Pydantic models ─────────────────────────────────

// Strategy labels returned by backend/ml/wine/serving/serve_cf.cf_strategy_name().
// Keep this union in sync with that function.
export type CfStrategy =
  | 'popularity_cold_start'  // user has no explicit wine ratings yet
  | 'wine_item_sim'          // wines use item-sim from history
  | 'none'

export interface WineScoreOut {
  wine_id:      number
  wine_name:    string
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
  variety:       string | null
  harmonize_csv: string | null
}

export interface WineSearchHit {
  id:            number
  name:          string
  style:         string | null
  harmonize_csv: string | null
  producer:      string | null
  abv:           number | null
  avg_rating:    number | null
  n_ratings:     number
}

export interface WineDetail {
  id: number
  name: string
  producer: string | null
  country: string | null
  abv: number | null
  avg_rating: number | null
  n_ratings: number
  style: string | null
  grapes_csv: string | null
  region: string | null
  body: string | null
  acidity: string | null
  harmonize_csv: string | null
}

// ── API helpers ─────────────────────────────────────────────────────────────

/** Path B — "For You" rankings (uses user history). */
export const getRankedWines = (userId: number, topN = 24) =>
  api.get<WineScoreOut[]>('/wine/ranked', {
    params: { user_id: userId, top_n: topN },
  }).then(r => r.data)

/** Path A — pair wines with a specific recipe. */
export const getWinePairings = (recipeId: number, userId: number, topN = 6) =>
  api.get<WineScoreOut[]>(`/wine/pairings/${recipeId}`, {
    params: { user_id: userId, top_n: topN },
  }).then(r => r.data)

export const searchWines = (q: string, limit = 40) =>
  api.get<WineSearchHit[]>('/wine/search', {
    params: { q, limit },
  }).then(r => r.data)

export const getWineDetail = (wineId: number) =>
  api.get<WineDetail>(`/wine/${wineId}`).then(r => r.data)

export const rateWine = (userId: number, wineId: number, rating: number) =>
  api.post<{ status: string; event_id: number }>('/wine-events', {
    user_id: userId,
    wine_id: wineId,
    event_type: 'rate',
    rating,
  }).then(r => r.data)
