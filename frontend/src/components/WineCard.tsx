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
  isRated?:   boolean
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
      <span style={{ fontSize: 11, color: 'var(--gray-500)', height: 14, display: 'block' }}>
        {selected > 0 ? `Rated: ${labels[selected]}` : hovered > 0 ? labels[hovered] : ''}
      </span>
    </div>
  )
}

// Per-style accent colors — single source of truth for both the card tint and
// the style picker chips. `bg` is a soft wash; `accent` the saturated border.
export const STYLE_COLORS: Record<string, { bg: string; accent: string }> = {
  // Pulled apart by HUE (not just shade) so adjacent styles are distinct:
  Red:            { bg: '#fbeef0', accent: '#9b1b30' },  // cool deep crimson
  White:          { bg: '#f4faf0', accent: '#9bbf4a' },  // straw / green-gold
  Rosé:           { bg: '#fdf2f4', accent: '#e8819f' },  // clear pink
  Sparkling:      { bg: '#fffce8', accent: '#f2c200' },  // bright champagne yellow
  Dessert:        { bg: '#f6ece0', accent: '#c8771a' },  // warm tawny orange-brown
  'Dessert/Port': { bg: '#f6ece0', accent: '#c8771a' },
}

// Subtle per-style tint for a card: soft background wash + matching accent
// border so a card visually reads as its style. Kept light for legibility.
export function styleTint(style: string | null): { background: string; borderLeft: string } {
  const c = style ? STYLE_COLORS[style] : undefined
  return c
    ? { background: c.bg, borderLeft: `4px solid ${c.accent}` }
    : { background: 'var(--surface, #fff)', borderLeft: '4px solid var(--gray-200)' }
}

// ── main card ───────────────────────────────────────────────────────────
export function WineCard({ wine, userId, isRated, onRated, onDismiss }: Props) {
  const [showStars,  setShowStars]  = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const handleRate = async (stars: number) => {
    setSubmitting(true)
    try {
      await rateWine(userId, wine.wine_id, stars)
      onRated?.()
    } finally {
      setSubmitting(false)
    }
  }

  const subtitle = [wine.style, wine.variety].filter(Boolean).join(' · ')

  return (
    <div className="card" style={{
      display: 'flex', flexDirection: 'column', gap: 0, height: '100%',
      position: 'relative',
      ...styleTint(wine.style),
      ...(isRated ? { opacity: 0.6 } : {}),
    }}>

      {/* Rated overlay — sits on top without shifting layout */}
      {isRated && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          background: 'rgba(255,255,255,0.55)', borderRadius: 'inherit', zIndex: 1,
          pointerEvents: 'none',
        }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--green-700)',
            background: 'var(--green-50)', border: '1px solid var(--green-200)',
            borderRadius: 20, padding: '3px 12px' }}>
            ✓ Rated
          </span>
        </div>
      )}

      {/* Header row */}
      <div style={{ marginBottom: 10 }}>
        <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4, lineHeight: 1.3 }}>
          {wine.wine_name}
        </h3>
        {wine.producer && (
          <p style={{ fontSize: 12, color: 'var(--gray-500)', marginBottom: 4 }}>
            {wine.producer}
          </p>
        )}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {subtitle && <span className="badge badge-gray">{subtitle}</span>}
          {Number.isFinite(wine.avg_rating) && (
            <span className="badge badge-amber">
              ★ {Number(wine.avg_rating).toFixed(1)}
              <span style={{ fontWeight: 400, opacity: .7 }}> ({wine.n_ratings})</span>
            </span>
          )}
        </div>
      </div>

      {/* Star rating — always in DOM to avoid layout shift; hidden until Rate clicked */}
      <div style={{
        marginTop: 8, padding: '8px 0',
        borderTop: '1px solid var(--gray-100)',
        borderBottom: '1px solid var(--gray-100)',
        visibility: showStars ? 'visible' : 'hidden',
      }}>
        <StarRating onRate={handleRate} submitting={submitting} />
      </div>

      {/* Action buttons — always in DOM, Rate button hidden once stars are showing */}
      <div style={{
        display: 'flex', gap: 8, marginTop: 'auto', paddingTop: 10,
        borderTop: '1px solid var(--gray-100)',
      }}>
        <button
          className="btn btn-primary"
          style={{ flex: 1, visibility: showStars ? 'hidden' : 'visible' }}
          onClick={() => setShowStars(true)}
          disabled={isRated}
        >
          ★ Rate
        </button>
        <button className="btn btn-ghost" onClick={onDismiss} disabled={isRated}>
          Dismiss
        </button>
      </div>
    </div>
  )
}
