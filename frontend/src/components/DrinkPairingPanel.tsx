// DrinkPairingPanel.tsx
// Path A — pairing suggestions for a single recipe.
// Mounted on RecipeDetailPage. Calls GET /drinks/pairings/{recipeId}.
//
// Visually denser than the Path-B feed so 4-6 picks fit on one screen.
// The "why" line surfaces the expert boost reason (Harmonize match for
// wines) when it's non-zero.

import { useCallback, useEffect, useState } from 'react'
import type { DrinkScoreOut, KindFilter } from '../api/drinks'
import { getDrinkPairings, rateDrink } from '../api/drinks'

interface Props {
  recipeId:        number
  recipeName:      string
  recipeTags?:     string[]
  userId:          number
  defaultKind?:    KindFilter
}

const KIND_OPTIONS: { key: KindFilter; label: string; icon: string }[] = [
  { key: 'wine', label: 'Wine', icon: '🍷' },
]

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
function pairingReason(d: DrinkScoreOut, recipeTags: string[]): string | null {
  if (d.expert_boost > 0) {
    if (d.kind === 'wine' && d.harmonize_csv) {
      // Find which harmonize tags overlap with the recipe context
      const harmonize = d.harmonize_csv.split(',').map(s => s.trim()).filter(Boolean)
      return `Harmonizes with ${harmonize.slice(0, 3).join(', ')}`
    }
    return 'Classic pairing'
  }
  // No expert hit — describe what won the slot
  if (d.cb_score >= d.cf_score && d.cb_score >= d.prior_score) {
    return 'Flavor match'
  }
  if (d.cf_score >= d.prior_score) {
    return 'Loved by similar drinkers'
  }
  return null   // pure popularity — no compelling story, leave blank
}

// ── compact pairing card ────────────────────────────────────────────────
function PairingCard({
  drink, userId, recipeTags,
}: {
  drink:      DrinkScoreOut
  userId:     number
  recipeTags: string[]
}) {
  const [phase,      setPhase]      = useState<'idle' | 'rated'>('idle')
  const [submitting, setSubmitting] = useState(false)

  const handleRate = async (stars: number) => {
    setSubmitting(true)
    try {
      await rateDrink(userId, drink.drink_id, stars)
      setPhase('rated')
    } finally {
      setSubmitting(false)
    }
  }

  const reason = pairingReason(drink, recipeTags)
  const icon   = '🍷'
  const sub    = [drink.style, drink.grapes_csv].filter(Boolean).join(' · ')

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
        <MiniRing score={drink.final_score} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{
            fontSize: 13, fontWeight: 600, color: 'var(--gray-900)',
            lineHeight: 1.25, margin: 0,
            overflow: 'hidden', display: '-webkit-box',
            WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
          }}>
            <span style={{ marginRight: 4 }}>{icon}</span>{drink.drink_name}
          </p>
          {drink.producer && (
            <p style={{ fontSize: 11, color: 'var(--gray-500)', margin: '2px 0 0' }}>
              {drink.producer}
            </p>
          )}
        </div>
      </div>

      {/* Subtitle / kind info */}
      {sub && (
        <p style={{ fontSize: 11, color: 'var(--gray-600)', margin: 0 }}>{sub}</p>
      )}

      {/* The "why" line — only when there's a story to tell */}
      {reason && (
        <p style={{
          fontSize: 11, color: drink.expert_boost > 0 ? 'var(--green-700)' : 'var(--gray-500)',
          margin: 0, fontWeight: drink.expert_boost > 0 ? 500 : 400,
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
          <span>{drink.expert_boost > 0 ? '🎯' : '✨'}</span>{reason}
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
            {drink.avg_rating != null && (
              <span style={{ fontSize: 10, color: 'var(--gray-400)' }}>
                avg ★ {drink.avg_rating.toFixed(1)}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── main panel ──────────────────────────────────────────────────────────
export function DrinkPairingPanel({
  recipeId, recipeName, recipeTags = [], userId, defaultKind = 'wine',
}: Props) {
  const [pairings, setPairings] = useState<DrinkScoreOut[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const [kind,     setKind]     = useState<KindFilter>(defaultKind)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const data = await getDrinkPairings(recipeId, userId, kind, 6)
      setPairings(data)
    } catch {
      setError('Could not load pairings.')
    } finally {
      setLoading(false)
    }
  }, [recipeId, userId, kind])

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
            Pair this with…
          </h2>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: '2px 0 0' }}>
            Suggestions for <em>{recipeName}</em>
          </p>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {KIND_OPTIONS.map(o => (
            <button
              key={o.key}
              onClick={() => setKind(o.key)}
              className={`badge ${kind === o.key ? 'badge-green' : 'badge-gray'}`}
              style={{
                cursor: 'pointer',
                padding: '4px 10px', fontSize: 12,
                border: kind === o.key ? '1px solid var(--green-500)' : '1px solid var(--gray-200)',
                transition: 'all .15s',
              }}
            >
              <span style={{ marginRight: 4 }}>{o.icon}</span>{o.label}
            </button>
          ))}
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
            No pairings available — make sure the drink models are trained.
          </p>
        </div>
      )}

      {!loading && !error && pairings.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: 12,
        }}>
          {pairings.map(d => (
            <PairingCard
              key={d.drink_id}
              drink={d}
              userId={userId}
              recipeTags={recipeTags}
            />
          ))}
        </div>
      )}
    </section>
  )
}
