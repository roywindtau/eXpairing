import { useEffect, useState, useCallback } from 'react'
import { getRankedRecipes } from '../api/client'
import type { RecipeScore } from '../api/client'
import { RecipeCard } from '../components/RecipeCard'

interface Props { userId: number }

type SortKey = 'final_score' | 'cf_score' | 'cb_score' | 'expiry_urgency' | 'match_ratio'

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: 'final_score',    label: 'Total score' },
  { key: 'cf_score',       label: 'CF score' },
  { key: 'cb_score',       label: 'CB score' },
  { key: 'expiry_urgency', label: 'Expiry urgency' },
  { key: 'match_ratio',    label: 'Pantry match' },
]

function CfStrategyBanner({ strategy }: { strategy: string | null }) {
  if (!strategy || strategy === 'none') return null
  const isCold = strategy === 'item_based_cold_start'
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '12px 16px', marginBottom: 16,
      background: isCold ? 'var(--blue-50)' : 'var(--green-50)',
      border: `1px solid ${isCold ? 'var(--blue-100)' : 'var(--green-200)'}`,
      borderRadius: 'var(--radius-md)', fontSize: 13,
      boxShadow: 'var(--shadow-sm)',
    }}>
      <span style={{
        fontSize: 18, width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
        background: 'rgba(255,255,255,.75)', border: '1px solid rgba(255,255,255,.9)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>{isCold ? '🌱' : '✨'}</span>
      <div>
        <span style={{ fontWeight: 600, color: isCold ? 'var(--blue-600)' : 'var(--green-700)' }}>
          {isCold ? 'Personalized for you (new user)' : 'Personalized from your history'}
        </span>
        <p style={{ fontSize: 12, color: isCold ? 'var(--blue-500)' : 'var(--green-600)', marginTop: 1 }}>
          {isCold
            ? 'Recommendations use community patterns + your diet preferences. Rate 5 recipes to unlock full personalization.'
            : 'Using your rating history with matrix factorization (CF model).'}
        </p>
      </div>
    </div>
  )
}

const PAGE_SIZE = 20
// Size of the default, MMR-diversified feed. Matches the backend's ranked
// pipeline (score → MMR rerank over top 3×N). The full unranked pool is only
// requested when the user searches, where MMR ordering is irrelevant.
const FEED_SIZE = 20

export function RecipeFeedPage({ userId }: Props) {
  const [recipes,  setRecipes]  = useState<RecipeScore[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const [strategy, setStrategy] = useState<string | null>(null)
  const [skipped,  setSkipped]  = useState<Set<number>>(new Set())
  const [cooked,   setCooked]   = useState<Set<number>>(new Set())
  const [sortKey,  setSortKey]  = useState<SortKey>('final_score')
  const [query,    setQuery]    = useState('')
  const [searchMode, setSearchMode] = useState(false)
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)

  // The default feed requests the MMR-diversified top-N (top_n>0) so the
  // backend's ranking → diversity rerank actually runs. Only when the user
  // searches do we pull the full scored pool (top_n=0, no MMR) to filter over.
  const load = useCallback(async (fullPool = false) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getRankedRecipes(userId, fullPool ? 0 : FEED_SIZE)
      setRecipes(data)
      setSearchMode(fullPool)
      if (data.length > 0) setStrategy(data[0].cf_strategy)
    } catch {
      setError('Could not load recipes. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }, [userId])

  useEffect(() => { load() }, [load])

  // The first non-empty search swaps the MMR feed for the full pool so there's
  // a real catalog to filter; clearing the box restores the diversified feed.
  useEffect(() => {
    if (query.trim() && !searchMode) load(true)
    else if (!query.trim() && searchMode) load(false)
  }, [query, searchMode, load])

  // Reset how many cards are shown whenever the search query changes.
  useEffect(() => { setVisibleCount(PAGE_SIZE) }, [query])

  const handleSkip   = (id: number) => setSkipped(prev => new Set([...prev, id]))
  const handleCooked = (id: number) => setCooked(prev => new Set([...prev, id]))

  const visible = recipes.filter(r => !skipped.has(r.recipe_id) && !cooked.has(r.recipe_id))
  // Always sort by the chosen key (highest first), including the default
  // "Total score" — the API pool isn't strictly ordered, so relying on its
  // order made the sort look inconsistent when toggling options.
  const sorted  = [...visible].sort((a, b) => b[sortKey] - a[sortKey])

  // Search filters by recipe name OR any ingredient (matched + missing = full list).
  const q = query.trim().toLowerCase()
  const filtered = q === ''
    ? sorted
    : sorted.filter(r =>
        r.recipe_name.toLowerCase().includes(q) ||
        [...r.matched_ingredients, ...r.missing_ingredients]
          .some(ing => ing.toLowerCase().includes(q))
      )

  const shown   = filtered.slice(0, visibleCount)
  const hasMore = filtered.length > visibleCount

  if (loading) return (
    <div className="page">
      <div className="spinner-wrap"><div className="spinner" /></div>
    </div>
  )

  if (error) return (
    <div className="page">
      <div className="empty">
        <div className="empty-icon">⚠️</div>
        <h3>Could not load recipes</h3>
        <p>{error}</p>
        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => load()}>
          Retry
        </button>
      </div>
    </div>
  )

  return (
    <div className="page">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h1 className="page-title" style={{ margin: 0 }}>What to cook tonight?</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ fontSize: 12, color: 'var(--gray-500)', whiteSpace: 'nowrap' }}>
            Sort by
          </label>
          <select
            value={sortKey}
            onChange={e => setSortKey(e.target.value as SortKey)}
            aria-label="Sort recipes by"
            style={{
              fontSize: 13, padding: '6px 10px',
              border: '1px solid var(--gray-300)', borderRadius: 8,
              background: 'white', color: 'var(--gray-700)',
              cursor: 'pointer', boxShadow: 'var(--shadow-sm)',
            }}
          >
            {SORT_OPTIONS.map(o => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
          </select>
          <button
            className="btn btn-ghost"
            onClick={() => { setSortKey('final_score'); setQuery(''); setSkipped(new Set()); setCooked(new Set()); load() }}
            style={{ fontSize: 13 }}
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      <CfStrategyBanner strategy={strategy} />

      {/* Search bar — narrows the loaded pool by recipe name or ingredient */}
      <div style={{ position: 'relative', marginBottom: 14 }}>
        <span style={{
          position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
          color: 'var(--gray-400)', pointerEvents: 'none',
        }}>🔍</span>
        <input
          className="form-input"
          placeholder="Search by name or ingredient…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          style={{ width: '100%', paddingLeft: 36 }}
        />
      </div>

      {visible.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">🍽️</div>
          <h3>No recipes to show</h3>
          <p>Add items to your pantry to get personalized recommendations.</p>
          <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => load()}>
            Refresh
          </button>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">🔍</div>
          <h3>No recipes found</h3>
          <p>Try a different search term.</p>
        </div>
      ) : (
        <>
          <p style={{ fontSize: 13, color: 'var(--gray-500)', marginBottom: 14 }}>
            {q
              ? `Showing ${shown.length} of ${filtered.length} recipes matching "${query.trim()}"`
              : `Showing ${shown.length} of ${filtered.length} recipes · ranked by CF · expiry urgency · pantry match`}
          </p>
          <div className="recipe-grid">
            {shown.map(r => (
              <RecipeCard
                key={r.recipe_id}
                recipe={r}
                userId={userId}
                onCooked={() => handleCooked(r.recipe_id)}
                onSkipped={() => handleSkip(r.recipe_id)}
              />
            ))}
          </div>
          {hasMore && (
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
              <button
                className="btn btn-ghost"
                onClick={() => setVisibleCount(c => c + PAGE_SIZE)}
              >
                Show more
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
