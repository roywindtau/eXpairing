// WinePairingPanel.tsx
// Path A — wine pairing suggestions for a single recipe.
// Mounted on RecipeDetailPage. Calls GET /wine/pairings/{recipeId}.
//
// Visually denser than the Path-B feed so 4-6 picks fit on one screen.
// The "why" line surfaces the expert boost reason (Harmonize match) when
// it's non-zero.

import { useCallback, useEffect, useState } from 'react'
import type { WineScoreOut } from '../api/wine'
import { getWinePairings, rateWine } from '../api/wine'

interface Props {
  recipeId:        number
  recipeName:      string
  recipeTags?:     string[]
  userId:          number
}

// ── compact score ring (36px) ───────────────────────────────────────────
function MiniRing({ score }: { score: number }) {
  const val   = Math.round(score * 100)
  const color = val >= 60 ? 'var(--blue-500)' : val >= 35 ? 'var(--blue-500)' : 'var(--gray-300)'
  const r     = 14
  const circ  = 2 * Math.PI * r
  const dash  = (val / 100) * circ
  return (
    <svg width={36} height={36} viewBox="0 0 36 36" style={{ flexShrink: 0 }}>
      <circle cx={18} cy={18} r={r} fill="none" stroke="var(--gray-100)" strokeWidth={3} />
      <circle cx={18} cy={18} r={r} fill="none" stroke={color} strokeWidth={3}
        strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
        transform="rotate(-90 18 18)" style={{ transition: 'stroke-dasharray .4s' }} />
      <text x={18} y={22} textAnchor="middle" fontSize={10} fontWeight={600} fill={color}>
        {val}
      </text>
    </svg>
  )
}

// ── 5-star compact input ────────────────────────────────────────────────
function MiniStars({ onRate, disabled }: { onRate: (n: number) => void; disabled: boolean }) {
  const [hovered,  setHovered]  = useState(0)
  const [selected, setSelected] = useState(0)
  const handleClick = (s: number) => { setSelected(s); onRate(s) }
  return (
    <div style={{ display: 'flex', gap: 2 }}>
      {[1, 2, 3, 4, 5].map(star => {
        const active = star <= (hovered || selected)
        return (
          <button
            key={star}
            disabled={disabled || selected > 0}
            onClick={() => handleClick(star)}
            onMouseEnter={() => setHovered(star)}
            onMouseLeave={() => setHovered(0)}
            style={{
              fontSize: 18, lineHeight: 1,
              color: active ? 'var(--amber-400)' : 'var(--gray-200)',
              cursor: selected > 0 ? 'default' : 'pointer',
              background: 'none', border: 'none', padding: '1px 2px',
              transition: 'color .1s',
            }}
          >★</button>
        )
      })}
    </div>
  )
}

// ── pairing reason text (the "why") ─────────────────────────────────────
// Surfaces the expert boost source when it's non-zero. Falls back to
// generic descriptors when CB/CF won without expert intervention.
function pairingReason(w: WineScoreOut): string | null {
  if (w.expert_boost > 0) {
    if (w.harmonize_csv) {
      // Find which harmonize tags overlap with the recipe context
      const harmonize = w.harmonize_csv.split(',').map(s => s.trim()).filter(Boolean)
      return `Harmonizes with ${harmonize.slice(0, 3).join(', ')}`
    }
    return 'Classic pairing'
  }
  // No expert hit — describe what won the slot
  if (w.cb_score >= w.cf_score && w.cb_score >= w.prior_score) {
    return 'Flavor match'
  }
  if (w.cf_score >= w.prior_score) {
    return 'Loved by similar drinkers'
  }
  return null   // pure popularity — no compelling story, leave blank
}

// ── compact pairing card ────────────────────────────────────────────────
function PairingCard({ wine, userId }: { wine: WineScoreOut; userId: number }) {
  const [phase,      setPhase]      = useState<'idle' | 'rated'>('idle')
  const [submitting, setSubmitting] = useState(false)

  const handleRate = async (stars: number) => {
    setSubmitting(true)
    try {
      await rateWine(userId, wine.wine_id, stars)
      setPhase('rated')
    } finally {
      setSubmitting(false)
    }
  }

  const reason = pairingReason(wine)
  const icon   = '🍷'
  const sub    = [wine.style, wine.variety].filter(Boolean).join(' · ')

  return (
    <div className="card" style={{
      display: 'flex', flexDirection: 'column', gap: 8,
      padding: 12,
      background: phase === 'rated' ? 'var(--green-50)' : undefined,
      borderColor: phase === 'rated' ? 'var(--green-100)' : undefined,
      transition: 'background .3s, border-color .3s',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <MiniRing score={wine.final_score} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{
            fontSize: 13, fontWeight: 600, color: 'var(--gray-900)',
            lineHeight: 1.25, margin: 0,
            overflow: 'hidden', display: '-webkit-box',
            WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
          }}>
            <span style={{ marginRight: 4 }}>{icon}</span>{wine.wine_name}
          </p>
          {wine.producer && (
            <p style={{ fontSize: 11, color: 'var(--gray-500)', margin: '2px 0 0' }}>
              {wine.producer}
            </p>
          )}
        </div>
      </div>

      {/* Subtitle info */}
      {sub && (
        <p style={{ fontSize: 11, color: 'var(--gray-600)', margin: 0 }}>{sub}</p>
      )}

      {/* The "why" line — only when there's a story to tell */}
      {reason && (
        <p style={{
          fontSize: 11, color: wine.expert_boost > 0 ? 'var(--green-700)' : 'var(--gray-500)',
          margin: 0, fontWeight: wine.expert_boost > 0 ? 500 : 400,
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
          <span>{wine.expert_boost > 0 ? '🎯' : '✨'}</span>{reason}
        </p>
      )}

      {/* Rating row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginTop: 'auto', paddingTop: 6,
        borderTop: '1px solid var(--gray-100)',
      }}>
        {phase === 'rated' ? (
          <span style={{ fontSize: 11, color: 'var(--green-700)', fontWeight: 500 }}>
            ✓ Rated
          </span>
        ) : (
          <>
            <MiniStars onRate={handleRate} disabled={submitting} />
            {wine.avg_rating != null && (
              <span style={{ fontSize: 10, color: 'var(--gray-400)' }}>
                avg ★ {wine.avg_rating.toFixed(1)}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── main panel ──────────────────────────────────────────────────────────
export function WinePairingPanel({ recipeId, recipeName, userId }: Props) {
  const [pairings, setPairings] = useState<WineScoreOut[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const data = await getWinePairings(recipeId, userId, 6)
      setPairings(data)
    } catch {
      setError('Could not load pairings.')
    } finally {
      setLoading(false)
    }
  }, [recipeId, userId])

  useEffect(() => { load() }, [load])

  return (
    <section style={{
      marginTop: 28, paddingTop: 24,
      borderTop: '1px solid var(--gray-100)',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 12, flexWrap: 'wrap', marginBottom: 14,
      }}>
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--gray-800)', margin: 0 }}>
            <span style={{ marginRight: 6 }}>🍷</span>Pair this with wine…
          </h2>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: '2px 0 0' }}>
            Suggestions for <em>{recipeName}</em>
          </p>
        </div>
      </div>

      {/* Body */}
      {loading && (
        <div className="spinner-wrap" style={{ minHeight: 120 }}>
          <div className="spinner" />
        </div>
      )}

      {!loading && error && (
        <div className="empty" style={{ padding: '20px 16px' }}>
          <p style={{ fontSize: 13, color: 'var(--gray-500)' }}>{error}</p>
          <button className="btn btn-ghost" style={{ marginTop: 10, fontSize: 12 }} onClick={load}>
            ↻ Retry
          </button>
        </div>
      )}

      {!loading && !error && pairings.length === 0 && (
        <div className="empty" style={{ padding: '20px 16px' }}>
          <p style={{ fontSize: 13, color: 'var(--gray-500)' }}>
            No pairings available — make sure the wine models are trained.
          </p>
        </div>
      )}

      {!loading && !error && pairings.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: 12,
        }}>
          {pairings.map(w => (
            <PairingCard key={w.wine_id} wine={w} userId={userId} />
          ))}
        </div>
      )}
    </section>
  )
}
