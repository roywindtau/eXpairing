// WineForYouPage.tsx
// "Suggest me a wine" feed.
// User clicks the button → GET /wine/ranked returns the top 10 popular wines.
// No auto-fetch, no personalization yet — the click IS the recommendation.

import { useCallback, useState } from 'react'
import { getRankedWines } from '../api/wine'
import type { WineOut } from '../api/wine'
import { WineCard } from '../components/WineCard'

interface Props { userId: number }

const SUGGEST_COUNT = 10

export function WineForYouPage({ userId }: Props) {
  const [wines,     setWines]     = useState<WineOut[]>([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [hasAsked,  setHasAsked]  = useState(false)
  const [dismissed, setDismissed] = useState<Set<number>>(new Set())

  const suggest = useCallback(async () => {
    setLoading(true)
    setError(null)
    setHasAsked(true)
    setDismissed(new Set())
    try {
      const data = await getRankedWines(SUGGEST_COUNT)
      setWines(data)
    } catch {
      setError('Could not get a suggestion. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleRated   = (id: number) =>
    setDismissed(prev => new Set([...prev, id]))
  const handleDismiss = (id: number) =>
    setDismissed(prev => new Set([...prev, id]))

  const visible = wines.filter(w => !dismissed.has(w.wine_id))

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
          <button className="btn btn-ghost" onClick={suggest} style={{ fontSize: 13 }}>
            ↻ Suggest again
          </button>
        )}
      </div>

      {/* Initial state — the call to action */}
      {!hasAsked && (
        <div className="empty" style={{ paddingTop: 48 }}>
          <div className="empty-icon">🥂</div>
          <h3>Not sure what to drink?</h3>
          <p style={{ marginBottom: 20 }}>
            Get {SUGGEST_COUNT} popular wine picks to get you started.
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
        visible.length === 0 ? (
          <div className="empty">
            <div className="empty-icon">🥂</div>
            <h3>No wines to show</h3>
            <p>{wines.length === 0
              ? 'The catalog returned nothing — check the backend and wine data.'
              : 'You\'ve gone through all the picks.'}
            </p>
            <div style={{ marginTop: 16 }}><SuggestButton label="Suggest again" /></div>
          </div>
        ) : (
          <>
            <p style={{ fontSize: 13, color: 'var(--gray-400)', marginBottom: 14 }}>
              {visible.length} popular picks
            </p>
            <div className="recipe-grid">
              {visible.map(w => (
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
        )
      )}
    </div>
  )
}
