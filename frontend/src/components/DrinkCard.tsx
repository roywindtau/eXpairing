// DrinkCard.tsx
// Compact card for a single drink in the "Drinks For You" feed.
// Two interactions:
//   Rate     -> POST /drink-events with 1-5 stars (feeds drink CF / item-sim)
//   Dismiss  -> client-side only; removes the card from view (no event)

import { useState } from 'react'
import type { DrinkScoreOut } from '../api/drinks'
import { rateDrink } from '../api/drinks'

interface Props {
  drink:      DrinkScoreOut
  userId:     number
  onRated?:   () => void
  onDismiss?: () => void
}

// ── score ring (mirrors RecipeCard's Ring) ──────────────────────────────
function ScoreRing({ score }: { score: number }) {
  const val   = Math.round(score * 100)
  const color = val >= 60 ? 'var(--blue-500)' : val >= 35 ? 'var(--blue-500)' : 'var(--gray-300)'
  const r     = 18
  const circ  = 2 * Math.PI * r
  const dash  = (val / 100) * circ
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <svg width={44} height={44} viewBox="0 0 44 44">
        <circle cx={22} cy={22} r={r} fill="none" stroke="var(--gray-100)" strokeWidth={4} />
        <circle cx={22} cy={22} r={r} fill="none" stroke={color} strokeWidth={4}
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 22 22)" style={{ transition: 'stroke-dasharray .4s' }} />
        <text x={22} y={27} textAnchor="middle" fontSize={11} fontWeight={600} fill={color}>
          {val}
        </text>
      </svg>
      <span style={{ fontSize: 10, color: 'var(--gray-500)' }}>score</span>
    </div>
  )
}

// ── horizontal mini bar for one score component ─────────────────────────
function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
      <span style={{ width: 50, color: 'var(--gray-500)' }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: 'var(--gray-100)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width .3s' }} />
      </div>
      <span style={{ width: 28, textAlign: 'right', color: 'var(--gray-600)', fontVariantNumeric: 'tabular-nums' }}>
        {pct}
      </span>
    </div>
  )
}

// ── inline star rating ──────────────────────────────────────────────────
function StarRating({
  onRate, submitting,
}: { onRate: (stars: number) => void; submitting: boolean }) {
  const [hovered,  setHovered]  = useState(0)
  const [selected, setSelected] = useState(0)

  const handleClick = (star: number) => {
    setSelected(star)
    onRate(star)
  }

  const labels = ['', 'Bad', 'Meh', 'OK', 'Good', 'Loved it']

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, padding: '4px 0' }}>
      <div style={{ display: 'flex', gap: 4 }}>
        {[1, 2, 3, 4, 5].map(star => {
          const active = star <= (hovered || selected)
          return (
            <button
              key={star}
              disabled={submitting || selected > 0}
              onClick={() => handleClick(star)}
              onMouseEnter={() => setHovered(star)}
              onMouseLeave={() => setHovered(0)}
              style={{
                fontSize: 24, lineHeight: 1,
                color: active ? 'var(--amber-400)' : 'var(--gray-200)',
                transform: active ? 'scale(1.12)' : 'scale(1)',
                transition: 'color .1s, transform .1s',
                cursor: selected > 0 ? 'default' : 'pointer',
                background: 'none', border: 'none', padding: '2px 3px',
              }}
            >★</button>
          )
        })}
      </div>
      {(hovered > 0 || selected > 0) && (
        <span style={{ fontSize: 11, color: 'var(--gray-500)', height: 14 }}>
          {selected > 0 ? `Rated: ${labels[selected]}` : labels[hovered]}
        </span>
      )}
    </div>
  )
}

// ── "why this drink" reason (Path B) ─────────────────────────────────────
// Translates the dominant score component into a short human sentence so the
// user understands the pick beyond just the algorithm name. No expert boost
// here (Path B doesn't have a specific recipe to apply expert rules against).
function whyForYou(drink: DrinkScoreOut): { text: string; icon: string } | null {
  const { cb_score, cf_score, prior_score, cf_strategy } = drink

  // Nothing won by a meaningful margin → no compelling story
  if (Math.max(cb_score, cf_score, prior_score) < 0.05) return null

  // CB dominates → flavor bridge picked it from the user's food history
  if (cb_score >= cf_score && cb_score >= prior_score) {
    return { icon: '🍽️', text: 'Matches your food taste' }
  }

  // CF dominates → describe HOW the CF score was computed
  if (cf_score >= prior_score) {
    switch (cf_strategy) {
      case 'biased_mf':
        return { icon: '✨', text: 'Predicted from your drink ratings' }
      case 'blended':
        return { icon: '✨', text: 'Blends your ratings with similar drinkers' }
      case 'wine_item_sim':
      case 'beer_item_sim':
        return { icon: '🤝', text: 'Similar to drinks you\'ve liked' }
      case 'popularity_cold_start':
        return { icon: '🔥', text: 'Loved by the community' }
      default:
        return { icon: '🤝', text: 'Picked from your history' }
    }
  }

  // Popularity prior won → just a generally well-loved drink
  return { icon: '🔥', text: 'Highly rated overall' }
}

// ── main card ───────────────────────────────────────────────────────────
export function DrinkCard({ drink, userId, onRated, onDismiss }: Props) {
  const [expanded,   setExpanded]   = useState(false)
  const [phase,      setPhase]      = useState<'idle' | 'rating' | 'rated'>('idle')
  const [submitting, setSubmitting] = useState(false)

  const handleRate = async (stars: number) => {
    setSubmitting(true)
    try {
      await rateDrink(userId, drink.drink_id, stars)
      setPhase('rated')
      setTimeout(() => onRated?.(), 800)
    } finally {
      setSubmitting(false)
    }
  }

  // Rated confirmation state
  if (phase === 'rated') {
    return (
      <div className="card" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: 120, background: 'var(--green-50)',
        border: '1px solid var(--green-100)',
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 28, marginBottom: 6 }}>✓</div>
          <p style={{ fontSize: 14, color: 'var(--green-700)', fontWeight: 500 }}>
            Rated!
          </p>
          <p style={{ fontSize: 12, color: 'var(--green-600)', marginTop: 2 }}>
            Your rating sharpens future drink picks.
          </p>
        </div>
      </div>
    )
  }

  const icon = drink.kind === 'beer' ? '🍺' : '🍷'
  const kindBadge = drink.kind === 'beer' ? 'badge-amber' : 'badge-green'
  // Pick the most informative subtitle line per kind
  const subtitle =
    drink.kind === 'beer'
      ? [drink.style, drink.abv ? `${drink.abv.toFixed(1)}% ABV` : null].filter(Boolean).join(' · ')
      : [drink.wine_type, drink.variety].filter(Boolean).join(' · ')

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
        <ScoreRing score={drink.final_score} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4, lineHeight: 1.3 }}>
            <span style={{ marginRight: 6 }}>{icon}</span>
            {drink.drink_name}
          </h3>
          {drink.producer && (
            <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>
              {drink.producer}
            </p>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            <span className={`badge ${kindBadge}`}>{drink.kind}</span>
            {subtitle && <span className="badge badge-gray">{subtitle}</span>}
            {drink.avg_rating != null && (
              <span className="badge badge-amber">
                ★ {drink.avg_rating.toFixed(1)}
                <span style={{ fontWeight: 400, opacity: .7 }}> ({drink.n_ratings})</span>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Pairing hint for wines */}
      {drink.kind === 'wine' && drink.harmonize_csv && (
        <p style={{ fontSize: 11, color: 'var(--gray-500)', marginBottom: 6 }}>
          Pairs with: {drink.harmonize_csv.replace(/,/g, ', ')}
        </p>
      )}

      {/* "Why this drink" — human translation of the dominant signal */}
      {(() => {
        const why = whyForYou(drink)
        return why ? (
          <p style={{
            fontSize: 11, color: 'var(--green-700)', fontWeight: 500,
            margin: '0 0 10px', display: 'flex', alignItems: 'center', gap: 4,
          }}>
            <span>{why.icon}</span>{why.text}
          </p>
        ) : null
      })()}

      {/* Expandable score breakdown */}
      <button
        onClick={() => setExpanded(e => !e)}
        style={{
          fontSize: 12, color: 'var(--blue-600)', background: 'none', border: 'none',
          textAlign: 'left', padding: 0, marginBottom: expanded ? 8 : 4,
        }}
      >
        {expanded ? '▲ Hide breakdown' : '▼ Why this drink?'}
      </button>
      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 10 }}>
          <ScoreBar label="Taste"  value={drink.cb_score}    color="var(--blue-500)"  />
          <ScoreBar label="Crowd"  value={drink.cf_score}    color="var(--green-500)" />
          {drink.expert_boost > 0 && (
            <ScoreBar label="Pairing" value={drink.expert_boost} color="var(--amber-400)" />
          )}
          <ScoreBar label="Pop."   value={drink.prior_score} color="var(--gray-400)" />
        </div>
      )}

      {/* Rating prompt (inline) */}
      {phase === 'rating' && (
        <div style={{
          marginTop: 8, padding: '8px 0',
          borderTop: '1px solid var(--gray-100)',
          borderBottom: '1px solid var(--gray-100)',
        }}>
          <StarRating onRate={handleRate} submitting={submitting} />
        </div>
      )}

      {/* Action buttons */}
      {phase === 'idle' && (
        <div style={{
          display: 'flex', gap: 8, marginTop: 10, paddingTop: 10,
          borderTop: '1px solid var(--gray-100)',
        }}>
          <button
            className="btn btn-primary"
            style={{ flex: 1 }}
            onClick={() => setPhase('rating')}
          >
            ★ Rate
          </button>
          <button className="btn btn-ghost" onClick={onDismiss}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  )
}
