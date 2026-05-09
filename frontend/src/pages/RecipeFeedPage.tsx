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
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '10px 16px', marginBottom: 16,
      background: isCold ? 'var(--blue-50)' : 'var(--green-50)',
      border: `1px solid ${isCold ? '#bfdbfe' : 'var(--green-100)'}`,
      borderRadius: 'var(--radius-md)', fontSize: 13,
    }}>
      <span style={{ fontSize: 18 }}>{isCold ? '🌱' : '✨'}</span>
      <div>
        <span style={{ fontWeight: 600, color: isCold ? 'var(--blue-600)' : 'var(--green-700)' }}>
          {isCold ? 'Personalized for you (new user)' : 'Personalized from your history'}
        </span>
        <p style={{ fontSize: 12, color: isCold ? '#3b82f6' : 'var(--green-600)', marginTop: 1 }}>
          {isCold
            ? 'Recommendations use community patterns + your diet preferences. Rate 5 recipes to unlock full personalization.'
            : 'Using your rating history with matrix factorization (CF model).'}
        </p>
      </div>
    </div>
  )
}

export function RecipeFeedPage({ userId }: Props) {
  const [recipes,  setRecipes]  = useState<RecipeScore[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const [strategy, setStrategy] = useState<string | null>(null)
  const [skipped,  setSkipped]  = useState<Set<number>>(new Set())
  const [cooked,   setCooked]   = useState<Set<number>>(new Set())
  const [sortKey,  setSortKey]  = useState<SortKey>('final_score')

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getRankedRecipes(userId, 20)
      setRecipes(data)
      if (data.length > 0) setStrategy(data[0].cf_strategy)
    } catch {
      setError('Could not load recipes. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }, [userId])

  useEffect(() => { load() }, [load])

  const handleSkip   = (id: number) => setSkipped(prev => new Set([...prev, id]))
  const handleCooked = (id: number) => setCooked(prev => new Set([...prev, id]))

  const visible = recipes.filter(r => !skipped.has(r.recipe_id) && !cooked.has(r.recipe_id))
  const sorted  = sortKey === 'final_score'
    ? visible
    : [...visible].sort((a, b) => b[sortKey] - a[sortKey])

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
        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={load}>
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
              fontSize: 12, padding: '4px 8px',
              border: '1px solid var(--gray-300)', borderRadius: 4,
              background: 'white', color: 'var(--gray-700)',
              cursor: 'pointer',
            }}
          >
            {SORT_OPTIONS.map(o => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
          </select>
          <button
            className="btn btn-ghost"
            onClick={() => { setSortKey('final_score'); setSkipped(new Set()); setCooked(new Set()); load() }}
            style={{ fontSize: 13 }}
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      <CfStrategyBanner strategy={strategy} />

      {visible.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">🍽️</div>
          <h3>No recipes to show</h3>
          <p>Add items to your pantry to get personalized recommendations.</p>
          <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={load}>
            Refresh
          </button>
        </div>
      ) : (
        <>
          <p style={{ fontSize: 13, color: 'var(--gray-400)', marginBottom: 14 }}>
            {visible.length} recipes ranked by collaborative filtering (CF) · expiry urgency · pantry match
          </p>
          <div className="recipe-grid">
            {sorted.map(r => (
              <RecipeCard
                key={r.recipe_id}
                recipe={r}
                userId={userId}
                onCooked={() => handleCooked(r.recipe_id)}
                onSkipped={() => handleSkip(r.recipe_id)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
