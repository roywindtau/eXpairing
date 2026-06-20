// WineForYouPage.tsx
// Path B — "Suggest me a wine" feed.
// User clicks the button → GET /wine/ranked returns the top 10 wines for them
// (CF + popularity model). No auto-fetch: the click IS the recommendation.

import { useCallback, useState } from 'react'
import { getRankedWines } from '../api/wine'
import type { WineScoreOut } from '../api/wine'
import { WineCard } from '../components/WineCard'

interface Props { userId: number }

const SUGGEST_COUNT = 10

type SortKey = 'final_score' | 'cb_score' | 'cf_score' | 'prior_score'

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: 'final_score', label: 'Total score' },
  { key: 'cb_score',    label: 'Taste match' },
  { key: 'cf_score',    label: 'Crowd score' },
  { key: 'prior_score', label: 'Popularity'  },
]

function CfStrategyBanner({ strategy }: { strategy: string | null }) {
  if (!strategy || strategy === 'none') return null
  // Cold: user has zero explicit wine ratings — signals come from food taste
  // (synthesizer + flavor bridge) + popularity.
  // Warm: user has rated wines directly — item-sim / blended / SVD active.
  const isCold = strategy === 'popularity_cold_start'
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
          {isCold
            ? 'Personalized from your food taste'
            : 'Personalized from your wine ratings'}
        </span>
        <p style={{ fontSize: 12, color: isCold ? '#3b82f6' : 'var(--green-600)', marginTop: 1 }}>
          {isCold
            ? 'Recommendations bridge from the recipes you rated. Rate a few wines to sharpen this further.'
            : 'Using your explicit wine ratings with item-similarity / matrix factorization.'}
        </p>
      </div>
    </div>
  )
}

export function WineForYouPage({ userId }: Props) {
  const [wines,     setWines]     = useState<WineScoreOut[]>([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [strategy,  setStrategy]  = useState<string | null>(null)
  const [hasAsked,  setHasAsked]  = useState(false)
  const [sortKey,   setSortKey]   = useState<SortKey>('final_score')
  const [dismissed, setDismissed] = useState<Set<number>>(new Set())

  const suggest = useCallback(async () => {
    setLoading(true)
    setError(null)
    setHasAsked(true)
    setDismissed(new Set())
    try {
      const data = await getRankedWines(userId, SUGGEST_COUNT)
      setWines(data)
      setStrategy(data.length > 0 ? data[0].cf_strategy : null)
    } catch {
      setError('Could not get a suggestion. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }, [userId])

  const handleRated   = (id: number) =>
    setDismissed(prev => new Set([...prev, id]))
  const handleDismiss = (id: number) =>
    setDismissed(prev => new Set([...prev, id]))

  const visible = wines.filter(w => !dismissed.has(w.wine_id))
  const sorted  = sortKey === 'final_score'
    ? visible
    : [...visible].sort((a, b) => b[sortKey] - a[sortKey])

  // ── Suggest button (shared across states) ──────────────────────────────
  const SuggestButton = ({ label }: { label: string }) => (
    <button
      className="btn btn-primary"
      onClick={suggest}
      disabled={loading}
      style={{ fontSize: 15, padding: '10px 22px' }}
    >
      🍷 {loading ? 'Finding wines…' : label}
    </button>
  )

  return (
    <div className="page">
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16, gap: 12, flexWrap: 'wrap',
      }}>
        <h1 className="page-title" style={{ margin: 0 }}>🍷 Wine for you</h1>
        {hasAsked && !loading && visible.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ fontSize: 12, color: 'var(--gray-500)', whiteSpace: 'nowrap' }}>
              Sort by
            </label>
            <select
              value={sortKey}
              onChange={e => setSortKey(e.target.value as SortKey)}
              aria-label="Sort wines by"
              style={{
                fontSize: 12, padding: '4px 8px',
                border: '1px solid var(--gray-300)', borderRadius: 4,
                background: 'white', color: 'var(--gray-700)', cursor: 'pointer',
              }}
            >
              {SORT_OPTIONS.map(o => (
                <option key={o.key} value={o.key}>{o.label}</option>
              ))}
            </select>
            <button className="btn btn-ghost" onClick={suggest} style={{ fontSize: 13 }}>
              ↻ Suggest again
            </button>
          </div>
        )}
      </div>

      {/* Initial state — the call to action */}
      {!hasAsked && (
        <div className="empty" style={{ paddingTop: 48 }}>
          <div className="empty-icon">🥂</div>
          <h3>Not sure what to drink?</h3>
          <p style={{ marginBottom: 20 }}>
            Get {SUGGEST_COUNT} wine picks tailored to you, ranked by taste, the
            crowd, and popularity.
          </p>
          <SuggestButton label="Suggest me a wine" />
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="spinner-wrap"><div className="spinner" /></div>
      )}

      {/* Error */}
      {hasAsked && !loading && error && (
        <div className="empty">
          <div className="empty-icon">⚠️</div>
          <h3>Could not get a suggestion</h3>
          <p>{error}</p>
          <div style={{ marginTop: 16 }}><SuggestButton label="Try again" /></div>
        </div>
      )}

      {/* Results */}
      {hasAsked && !loading && !error && (
        <>
          <CfStrategyBanner strategy={strategy} />
          {visible.length === 0 ? (
            <div className="empty">
              <div className="empty-icon">🥂</div>
              <h3>No wines to show</h3>
              <p>{wines.length === 0
                ? 'The model returned nothing — check the backend and wine data.'
                : 'You\'ve gone through all the picks.'}
              </p>
              <div style={{ marginTop: 16 }}><SuggestButton label="Suggest again" /></div>
            </div>
          ) : (
            <>
              <p style={{ fontSize: 13, color: 'var(--gray-400)', marginBottom: 14 }}>
                {visible.length} picks ranked by taste (CB) · crowd (CF) · popularity
              </p>
              <div className="recipe-grid">
                {sorted.map(w => (
                  <WineCard
                    key={w.wine_id}
                    wine={w}
                    userId={userId}
                    onRated={()   => handleRated(w.wine_id)}
                    onDismiss={() => handleDismiss(w.wine_id)}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
