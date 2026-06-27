// WineForYouPage.tsx
// "Suggest me a wine" feed.
// User clicks the button → GET /wine/ranked?user_id= returns a personalized
// ranking (style-filtered, CF+CB blend for warm users; popularity cold start).
// No auto-fetch — the click IS the recommendation.

import { useCallback, useState } from 'react'
import { getRankedWines } from '../api/wine'
import type { WineOut } from '../api/wine'
import { WineCard, STYLE_COLORS } from '../components/WineCard'

interface Props { userId: number }

const SUGGEST_COUNT = 10
const STYLE_OPTIONS = ['Red', 'White', 'Rosé', 'Sparkling', 'Dessert'] as const

export function WineForYouPage({ userId }: Props) {
  const [wines,     setWines]     = useState<WineOut[]>([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [hasAsked,  setHasAsked]  = useState(false)
  const [dismissed, setDismissed] = useState<Set<number>>(new Set())
  const [rated,     setRated]     = useState<Set<number>>(new Set())
  const [styles,    setStyles]    = useState<Set<string>>(new Set())

  const toggleStyle = (s: string) =>
    setStyles(prev => {
      const next = new Set(prev)
      next.has(s) ? next.delete(s) : next.add(s)
      return next
    })

  const suggest = useCallback(async () => {
    setLoading(true)
    setError(null)
    setHasAsked(true)
    setDismissed(new Set())
    setRated(new Set())
    try {
      const data = await getRankedWines(SUGGEST_COUNT, userId, [...styles])
      setWines(data)
    } catch {
      setError('Could not get a suggestion. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }, [userId, styles])

  const handleRated   = (id: number) =>
    setRated(prev => new Set([...prev, id]))
  const handleDismiss = (id: number) =>
    setDismissed(prev => new Set([...prev, id]))

  const visible = wines.filter(w => !dismissed.has(w.wine_id))

  // Group the feed into one section per style, preserving rank order within
  // each style and ordering styles by their best-ranked wine.
  const groupedByStyle: [string, WineOut[]][] = (() => {
    const groups = new Map<string, WineOut[]>()
    for (const w of visible) {
      const key = w.style ?? 'Other'
      if (!groups.has(key)) groups.set(key, [])
      groups.get(key)!.push(w)
    }
    return [...groups.entries()]   // insertion order = first appearance = best rank
  })()

  // Within a style row, sub-group wines by their PRIMARY pairing — the pairing
  // (from harmonize_csv) that is most common across that row, so cards cluster
  // into meaningful pairing groups instead of singletons.
  const groupByPairing = (row: WineOut[]): [string, WineOut[]][] => {
    const foods = (w: WineOut) =>
      (w.harmonize_csv ?? '').split(',').map(s => s.trim()).filter(Boolean)
    // global frequency of each food across the row
    const freq = new Map<string, number>()
    row.forEach(w => foods(w).forEach(f => freq.set(f, (freq.get(f) ?? 0) + 1)))
    const primary = (w: WineOut): string => {
      const fs = foods(w)
      if (!fs.length) return 'Other'
      return fs.reduce((a, b) => (freq.get(b)! > freq.get(a)! ? b : a))
    }
    const groups = new Map<string, WineOut[]>()
    for (const w of row) {
      const key = primary(w)
      if (!groups.has(key)) groups.set(key, [])
      groups.get(key)!.push(w)
    }
    // order pairing groups by size (biggest cluster first)
    return [...groups.entries()].sort((a, b) => b[1].length - a[1].length)
  }

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

  // Style chips — toggle which styles to generate. Empty = "styles you drink".
  const StylePicker = () => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, marginBottom: 18 }}>
      <span style={{ fontSize: 12, color: 'var(--gray-400)' }}>
        Styles {styles.size === 0 && '(all you drink)'}
      </span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center' }}>
        {STYLE_OPTIONS.map(s => {
          const on = styles.has(s)
          const c = STYLE_COLORS[s]
          return (
            <button
              key={s}
              type="button"
              onClick={() => toggleStyle(s)}
              className="badge"
              style={{
                cursor: 'pointer', fontSize: 12, padding: '4px 10px',
                // selected: filled with the style's accent; idle: soft wash
                background: on ? c.accent : c.bg,
                color: on ? '#fff' : 'var(--gray-700)',
                border: `1px solid ${c.accent}`,
                fontWeight: on ? 600 : 400,
              }}
            >
              {on ? '✓ ' : ''}{s}
            </button>
          )
        })}
      </div>
    </div>
  )

  return (
    <div className="page">
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16, gap: 12, flexWrap: 'wrap',
      }}>
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
            Get {SUGGEST_COUNT} wine picks tailored to what you've rated.
          </p>
          <StylePicker />
          <SuggestButton label="Suggest me a wine" />
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          gap: 12, paddingTop: 64,
        }}>
          <span className="wine-spinner" role="img" aria-label="pouring wine">🍷</span>
          <span style={{ fontSize: 13, color: 'var(--gray-400)' }}>Finding wines…</span>
        </div>
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
            <StylePicker />
            {groupedByStyle.map(([style, group]) => (
              <section key={style} style={{ marginBottom: 24 }}>
                <h2 style={{
                  fontSize: 14, fontWeight: 600, color: 'var(--gray-600)',
                  margin: '0 0 10px', display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  {style}
                  <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--gray-400)' }}>
                    {group.length}
                  </span>
                </h2>
                {groupByPairing(group).map(([pairing, wines]) => (
                  <div key={pairing} style={{ marginBottom: 14 }}>
                    <p style={{
                      fontSize: 12, color: 'var(--gray-400)', margin: '0 0 6px',
                      fontWeight: 500,
                    }}>
                      Pairs with {pairing}
                    </p>
                    <div className="wine-grid">
                      {wines.map(w => (
                        <WineCard
                          key={w.wine_id}
                          wine={w}
                          userId={userId}
                          isRated={rated.has(w.wine_id)}
                          onRated={()   => handleRated(w.wine_id)}
                          onDismiss={() => handleDismiss(w.wine_id)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </section>
            ))}
          </>
        )
      )}
    </div>
  )
}
