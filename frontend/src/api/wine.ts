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
  country:       string | null
  style:         string | null
  variety:       string | null
  harmonize_csv: string | null
  acidity:       string | null
  body:          string | null
  region:        string | null
}

// a wine paired to a recipe, with its cosine pairing score in [0, 1]
export interface PairedWine extends WineOut {
  pairing_score: number
}

// wine taste details inferred from the user's fruit picks (cold-start onboarding)
export interface WinePreferences {
  fruits:  string[]
  grapes:  string[]
  body:    string | null
  acidity: string | null
  styles:  string[]
}

// everyday fruits offered in onboarding — must match FRUIT_PROFILES keys
// (backend/services/wine/preference_profile.py)
export const FRUIT_OPTIONS = [
  'orange', 'lemon', 'grapes', 'apple', 'pear', 'peach',
  'apricot', 'cherry', 'strawberry', 'raspberry', 'blackberry', 'plum',
] as const

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

/**
 * "Pair me a wine for this recipe." Pure content-based: ranks wines by how well
 * their food-pairing profile matches the recipe's ingredients. No user history.
 */
export const pairWines = (recipeId: number, topN = 8) =>
  api.post<PairedWine[]>('/wine/pair', {
    recipe_id: recipeId,
    top_n: topN,
  }).then(r => r.data)

export const rateWine = (userId: number, wineId: number, rating: number) =>
  api.post<{ status: string; event_id: number }>('/wine-events', {
    user_id: userId,
    wine_id: wineId,
    event_type: 'rate',
    rating,
  }).then(r => r.data)

/** Cold-start onboarding: save the fruits a user enjoys; backend infers + stores
 * the wine taste details and returns them. */
export const saveWinePreferences = (userId: number, fruits: string[]) =>
  api.post<WinePreferences>('/wine/preferences', {
    user_id: userId,
    fruits,
  }).then(r => r.data)

/** Read a user's stored wine taste prefs (to prefill the picker). */
export const getWinePreferences = (userId: number) =>
  api.get<WinePreferences>('/wine/preferences', {
    params: { user_id: userId },
  }).then(r => r.data)
