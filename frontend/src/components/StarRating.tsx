import { useState } from 'react'

interface Props {
  onRate: (stars: number) => void
  submitting: boolean
}

// Renders 5 clickable stars. Hover previews, click confirms.
// Generates event_type=rate which feeds SVD training.
export function StarRating({ onRate, submitting }: Props) {
  const [hovered, setHovered] = useState(0)
  const [selected, setSelected] = useState(0)

  const handleClick = (star: number) => {
    setSelected(star)
    onRate(star)
  }

  const labels = ['', 'Terrible', 'Bad', 'OK', 'Good', 'Excellent']

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, padding: '12px 0' }}>
      <p style={{ fontSize: 13, color: 'var(--gray-600)', fontWeight: 500 }}>
        How was it? Your rating improves future recommendations.
      </p>
      <div style={{ display: 'flex', gap: 6 }}>
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
                fontSize: 28,
                lineHeight: 1,
                color: active ? 'var(--amber-400)' : 'var(--gray-200)',
                transition: 'color .1s, transform .1s',
                transform: active ? 'scale(1.15)' : 'scale(1)',
                cursor: selected > 0 ? 'default' : 'pointer',
                background: 'none',
                border: 'none',
                padding: '2px 4px',
              }}
            >
              ★
            </button>
          )
        })}
      </div>
      {(hovered > 0 || selected > 0) && (
        <span style={{ fontSize: 12, color: 'var(--gray-500)', height: 16 }}>
          {selected > 0 ? `Rated: ${labels[selected]}` : labels[hovered]}
        </span>
      )}
      {selected === 0 && (
        <button
          onClick={() => onRate(0)}
          style={{ fontSize: 12, color: 'var(--gray-400)', background: 'none', border: 'none', cursor: 'pointer' }}
        >
          Skip rating
        </button>
      )}
    </div>
  )
}
