import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
})

export default api

// ── types matching backend Pydantic models ─────────────────────────────────

export interface RecipeDetail {
  id: number
  name: string
  ingredients: string[]
  tags: string[]
  minutes: number | null
  n_steps: number | null
  avg_rating: number | null
  n_ratings: number
  description: string | null
  steps: string[]
}

export interface PantryItem {
  id: number
  ingredient: string
  expiry_date: string   // ISO "YYYY-MM-DD"
  raw_name: string | null
  quantity: string | null
}

export interface RecipeScore {
  recipe_id: number
  recipe_name: string
  final_score: number
  match_ratio: number
  expiry_urgency: number
  cf_score: number
  cb_score: number
  matched_ingredients: string[]
  missing_ingredients: string[]
  total_ingredients: number
  tags: string[]
  minutes: number | null
  avg_rating: number | null
  cf_strategy: 'biased_mf' | 'item_based_cold_start' | 'blended' | 'none'
  cb_model_available: boolean   // false → model not loaded; true → model loaded (cb_score=0 means genuine 0 similarity)
}

export interface UserProfile {
  id: number
  name: string | null
  beta: number
  has_cf: boolean
  has_cb: boolean
  diet_tags: string | null
}

// ── API helpers ─────────────────────────────────────────────────────────────

export const getPantry = (userId: number) =>
  api.get<PantryItem[]>(`/pantry/${userId}`).then(r => r.data)

export const addPantryItem = (userId: number, item: Omit<PantryItem, 'id'>) =>
  api.post<PantryItem>(`/pantry/${userId}`, item).then(r => r.data)

export const addPantryItemsBulk = (userId: number, items: Omit<PantryItem, 'id'>[]) =>
  api.post<PantryItem[]>(`/pantry/${userId}/bulk`, items).then(r => r.data)

export const deletePantryItem = (userId: number, itemId: number) =>
  api.delete(`/pantry/${userId}/${itemId}`)

export const getRankedRecipes = (userId: number, topN = 20) =>
  api.get<RecipeScore[]>(`/recipes/ranked`, {
    params: { user_id: userId, top_n: topN },
  }).then(r => r.data)

export const logEvent = (payload: {
  user_id: number
  recipe_id: number
  event_type: 'cook' | 'skip' | 'rate'
  rating?: number
  n_missing?: number
}) => api.post('/events', payload).then(r => r.data)

export const getRecipeDetail = (recipeId: number) =>
  api.get<RecipeDetail>(`/recipes/${recipeId}`).then(r => r.data)

export const getUser = (userId: number) =>
  api.get<UserProfile>(`/users/${userId}`).then(r => r.data)

export const createUser = (payload: {
  name?: string
  beta: number
  diet_tags?: string
}) => api.post<UserProfile>('/users', payload).then(r => r.data)

export const updateUser = (
  userId: number,
  payload: { name?: string; beta: number; diet_tags?: string },
) => api.put<UserProfile>(`/users/${userId}`, payload).then(r => r.data)

// ── Shopping list ───────────────────────────────────────────────────────────

export interface ShoppingItem {
  id: number
  ingredient: string
  source_recipe_id: number | null
  source_recipe_name: string | null
  is_checked: boolean
}

export const getShoppingList = (userId: number) =>
  api.get<ShoppingItem[]>(`/shopping/${userId}`).then(r => r.data)

export const addToShoppingList = (userId: number, payload: {
  ingredients: string[]
  recipe_id?: number
  recipe_name?: string
}) => api.post<{ added: ShoppingItem[]; skipped: string[] }>(
  `/shopping/${userId}`, payload
).then(r => r.data)

export const toggleShoppingItem = (userId: number, itemId: number, isChecked: boolean) =>
  api.patch<ShoppingItem>(`/shopping/${userId}/${itemId}`, { is_checked: isChecked }).then(r => r.data)

export const removeShoppingItem = (userId: number, itemId: number) =>
  api.delete(`/shopping/${userId}/${itemId}`)

export const clearShoppingList = (userId: number, checkedOnly = true) =>
  api.delete(`/shopping/${userId}`, { params: { checked_only: checkedOnly } })
