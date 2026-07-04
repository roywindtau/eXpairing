// WineForYouPage.tsx
// "Suggest me a wine" feed.
// User clicks the button → GET /wine/ranked?user_id= returns a personalized
// ranking (style-filtered, CF+CB blend for warm users; popularity cold start).
// No auto-fetch — the click IS the recommendation.

import { useCallback, useEffect, useState } from 'react'
import { getRankedWines, getWinePreferences, saveWinePreferences, FRUIT_OPTIONS } from '../api/wine'
import type { WineOut } from '../api/wine'
import { WineCard, STYLE_COLORS } from '../components/WineCard'

interface Props { userId: number }

const SUGGEST_COUNT = 10
const STYLE_OPTIONS = ['Red', 'White', 'Rosé', 'Sparkling', 'Dessert'] as const
const FRUIT_EMOJI: Record<string, string> = {
  orange: '🍊', lemon: '🍋', grapes: '🍇', apple: '🍎', pear: '🍐', peach: '🍑',
  apricot: '🟠', cherry: '🍒', strawberry: '🍓', raspberry: '🔴', blackberry: '⚫', plum: '🟣',
}

export function WineForYouPage({ userId }: Props) {
  const [wines,     setWines]     = useState<WineOut[]>([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [hasAsked,  setHasAsked]  = useState(false)
  const [dismissed, setDismissed] = useState<Set<number>>(new Set())
  const [rated,     setRated]     = useState<Set<number>>(new Set())
  const [styles,    setStyles]    = useState<Set<string>>(new Set())
  const [fruits,    setFruits]    = useState<Set<string>>(new Set())
  // Fruit picking is a ONE-TIME cold-start onboarding step. Once the user has
  // saved fruit prefs, we never show the picker again (not on results, not on
  // later visits) — taste then comes from the wines they rate.
  const [onboarded, setOnboarded] = useState(false)

  // Prefill the fruit picker from any previously-saved preferences.
  useEffect(() => {
    getWinePreferences(userId)
      .then(p => {
        setFruits(new Set(p.fruits))
        if (p.fruits.length > 0) setOnboarded(true)   // already onboarded
      })
      .catch(() => { /* no prefs yet / backend down — leave empty */ })
  }, [userId])

  const toggleStyle = (s: string) =>
    setStyles(prev => {
      const next = new Set(prev)
      next.has(s) ? next.delete(s) : next.add(s)
      return next
    })

  const toggleFruit = (f: string) =>
    setFruits(prev => {
      const next = new Set(prev)
      next.has(f) ? next.delete(f) : next.add(f)
      return next
    })

  const suggest = useCallback(async () => {
    setLoading(true)
    setError(null)
    setHasAsked(true)
    setDismissed(new Set())
    setRated(new Set())
    try {
      // Persist fruit picks first so the backend can seed the cold-start ranking.
      // Only when something is selected: avoids wiping stored prefs if the
      // mount-time load failed, and avoids needless writes on every click.
      if (fruits.size > 0) {
        await saveWinePreferences(userId, [...fruits])
        setOnboarded(true)   // picker won't reappear after this first pick
      }
      const data = await getRankedWines(SUGGEST_COUNT, userId, [...styles])
      setWines(data)
    } catch {
      setError('Could not get a suggestion. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }, [userId, styles, fruits])

  const handleRated   = (id: number) =>
    setRated(prev => new Set([...prev, id]))
  const handleDismiss = (id: number) =>
    setDismissed(prev => new Set([...prev, id]))

  const visible = wines.filter(w => !dismissed.has(w.wine_id))

  const SuggestButton = ({ label }: { label: string }) => (
    <button
      className="btn btn-primary"
      onClick={suggest}
      disabled={loading}
      style={{ fontSize: 19, padding: '17px 38px', borderRadius: 14 }}
    >
      🍷 {loading ? 'Finding wines…' : label}
    </button>
  )

  // Fruit chips — cold-start onboarding. Picks are inferred into a wine taste
  // profile on the backend and seed recommendations until the user rates wines.
  const FruitPicker = () => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, marginBottom: 36 }}>
      <span style={{ fontSize: 16, color: 'var(--gray-600)', fontWeight: 600 }}>
        Fruits you enjoy {fruits.size === 0 && '(helps us start)'}
      </span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', maxWidth: 640 }}>
        {FRUIT_OPTIONS.map(f => {
          const on = fruits.has(f)
          return (
            <button
              key={f}
              type="button"
              onClick={() => toggleFruit(f)}
              className={`badge pill${on ? ' is-on' : ''}`}
              style={{
                fontSize: 17, padding: '12px 22px', gap: '.5rem',
                textTransform: 'capitalize',
                background: on ? 'var(--purple-600, #7c3aed)' : 'var(--surface)',
                color: on ? '#fff' : 'var(--gray-700)',
                border: `1px solid ${on ? 'var(--purple-600, #7c3aed)' : 'var(--gray-300)'}`,
                fontWeight: on ? 600 : 500,
              }}
            >
              {FRUIT_EMOJI[f] ?? '🍇'} {on ? '✓ ' : ''}{f}
            </button>
          )
        })}
      </div>
    </div>
  )

  // Style chips — toggle which styles to generate. Empty = "styles you drink".
  const StylePicker = () => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, marginBottom: 36 }}>
      <span style={{ fontSize: 16, color: 'var(--gray-600)', fontWeight: 600 }}>
        Styles {styles.size === 0 && '(all you drink)'}
      </span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center' }}>
        {STYLE_OPTIONS.map(s => {
          const on = styles.has(s)
          const c = STYLE_COLORS[s]
          return (
            <button
              key={s}
              type="button"
              onClick={() => toggleStyle(s)}
              className={`badge pill${on ? ' is-on' : ''}`}
              style={{
                fontSize: 17, padding: '12px 24px', gap: '.5rem',
                // selected: filled with the style's accent; idle: soft wash
                background: on ? c.accent : c.bg,
                color: on ? '#fff' : 'var(--gray-700)',
                border: `1px solid ${c.accent}`,
                fontWeight: on ? 600 : 500,
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
        <div className="empty" style={{ paddingTop: 40 }}>
          <div className="empty-icon" style={{ width: 112, height: 112, fontSize: '3.1rem', marginBottom: '1.6rem' }}>🥂</div>
          <h3 style={{ fontSize: '2rem' }}>Not sure what to drink?</h3>
          <p style={{ marginBottom: 40, fontSize: '1.15rem', maxWidth: 560, marginLeft: 'auto', marginRight: 'auto' }}>
            {onboarded
              ? "Pick the styles you're after, or just hit suggest."
              : "New here? Tell us which fruits you enjoy and we'll start your picks. They sharpen as you rate wines."}
          </p>
          {!onboarded && <FruitPicker />}
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
            <p style={{
              fontSize: 11, color: 'var(--gray-500)', margin: '0 0 12px',
              fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.08em',
            }}>
              Picked for you
            </p>
            <div className="wine-grid">
              {visible.map(w => (
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
          </>
        )
      )}
    </div>
  )
}
