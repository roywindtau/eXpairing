// WineCard.tsx
// Compact card for a single wine in the "Suggest me a wine" feed.
// Two interactions:
//   Rate     -> POST /wine-events with 1-5 stars
//   Dismiss  -> client-side only; removes the card from view (no event)

import { useState } from 'react'
import type { WineOut } from '../api/wine'
import { rateWine } from '../api/wine'

interface Props {
  wine:       WineOut
  userId:     number
  onRated?:   () => void
  onDismiss?: () => void
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

// ── main card ───────────────────────────────────────────────────────────
export function WineCard({ wine, userId, onRated, onDismiss }: Props) {
  const [phase,      setPhase]      = useState<'idle' | 'rating' | 'rated'>('idle')
  const [submitting, setSubmitting] = useState(false)

  const handleRate = async (stars: number) => {
    setSubmitting(true)
    try {
      await rateWine(userId, wine.wine_id, stars)
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
        </div>
      </div>
    )
  }

  // Most informative subtitle line: wine style + grape variety
  const subtitle = [wine.style, wine.variety].filter(Boolean).join(' · ')

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>

      {/* Header row */}
      <div style={{ marginBottom: 10 }}>
        <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4, lineHeight: 1.3 }}>
          <span style={{ marginRight: 6 }}>🍷</span>
          {wine.wine_name}
        </h3>
        {wine.producer && (
          <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>
            {wine.producer}
          </p>
        )}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          <span className="badge badge-green">wine</span>
          {subtitle && <span className="badge badge-gray">{subtitle}</span>}
          {wine.avg_rating != null && (
            <span className="badge badge-amber">
              ★ {wine.avg_rating.toFixed(1)}
              <span style={{ fontWeight: 400, opacity: .7 }}> ({wine.n_ratings})</span>
            </span>
          )}
        </div>
      </div>

      {/* Pairing hint */}
      {wine.harmonize_csv && (
        <p style={{ fontSize: 11, color: 'var(--gray-500)', marginBottom: 6 }}>
          Pairs with: {wine.harmonize_csv.replace(/,/g, ', ')}
        </p>
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
