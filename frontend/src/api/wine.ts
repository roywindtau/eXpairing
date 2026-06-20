// wine.ts
// Typed wrappers around the backend wine endpoints.

import api from './client'

// ── types matching backend Pydantic models ─────────────────────────────────

export interface WineOut {
  wine_id:       number
  wine_name:     string
  avg_rating:    number | null
  n_ratings:     number
  abv:           number | null
  producer:      string | null
  style:         string | null
  variety:       string | null
  harmonize_csv: string | null
  acidity:       string | null
  body:          string | null
  region:        string | null
}

// ── API helpers ─────────────────────────────────────────────────────────────

/**
 * "Suggest me a wine" — top-N wines.
 * With userId: personalized (style-filtered, CF+CB blend, popularity cold start).
 * Without userId: top-N popular (back-compat).
 */
export const getRankedWines = (topN = 10, userId?: number, styles?: string[]) =>
  api.get<WineOut[]>('/wine/ranked', {
    params: {
      top_n: topN,
      ...(userId != null ? { user_id: userId } : {}),
      ...(styles && styles.length ? { styles } : {}),
    },
    paramsSerializer: { indexes: null },   // styles=Red&styles=White
  }).then(r => r.data)

export const rateWine = (userId: number, wineId: number, rating: number) =>
  api.post<{ status: string; event_id: number }>('/wine-events', {
    user_id: userId,
    wine_id: wineId,
    event_type: 'rate',
    rating,
  }).then(r => r.data)
