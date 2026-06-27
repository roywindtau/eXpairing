// WinePairing.tsx
// "Pair me a wine for this recipe" — a self-contained section that fetches and
// renders content-based wine pairings for a recipe. No user history involved.

import { useEffect, useState } from 'react'
import { pairWines } from '../api/wine'
import type { PairedWine } from '../api/wine'

interface Props {
  recipeId: number
  topN?: number
}

export function WinePairing({ recipeId, topN = 8 }: Props) {
  const [pairs, setPairs] = useState<PairedWine[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // reset when switching recipes
  useEffect(() => {
    setPairs(null)
    setError(null)
  }, [recipeId])

  const handlePair = () => {
    setLoading(true)
    setError(null)
    pairWines(recipeId, topN)
      .then(setPairs)
      .catch(() => setError('Could not load pairings. Try again.'))
      .finally(() => setLoading(false))
  }

  return (
    <section style={{ marginBottom: 28 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--gray-800)', margin: 0 }}>
          🍷 Wine pairing
        </h2>
        <button
          className="btn btn-primary"
          style={{ fontSize: 13 }}
          onClick={handlePair}
          disabled={loading}
        >
          {loading ? 'Finding…' : pairs ? 'Refresh' : 'Pair me a wine'}
        </button>
      </div>

      {error && (
        <p style={{ fontSize: 13, color: 'var(--red-500, #dc2626)' }}>{error}</p>
      )}

      {pairs && pairs.length === 0 && !loading && (
        <p style={{ fontSize: 14, color: 'var(--gray-400)' }}>
          No confident pairing found for this recipe.
        </p>
      )}

      {pairs && pairs.length > 0 && (
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {pairs.map(w => (
            <li key={w.wine_id} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              gap: 12, padding: '10px 14px', borderRadius: 10,
              border: '1px solid var(--gray-200)', background: 'var(--gray-50, #fafafa)',
            }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--gray-900)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {w.wine_name}
                </div>
                <div style={{ fontSize: 12, color: 'var(--gray-500)', marginTop: 2 }}>
                  {[w.style, w.variety].filter(Boolean).join(' · ')}
                  {w.harmonize_csv && <span> · pairs with {w.harmonize_csv}</span>}
                </div>
              </div>
              <span className="badge badge-amber" style={{ fontSize: 12, flexShrink: 0 }}>
                {Math.round(w.pairing_score * 100)}% match
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
