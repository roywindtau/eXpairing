// RecipeCard.tsx
// Shows a recipe with match ring, score explainer, and three interaction modes:
//   Cook  -> logs event_type=cook, reveals star rating input
//   Rate  -> logs event_type=rate with 1-5 stars (feeds SVD training)
//   Skip  -> logs event_type=skip, hides card from feed

import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { RecipeScore } from '../api/client'
import { ScoreExplainer } from './ScoreExplainer'
import { logEvent, addToShoppingList } from '../api/client'

interface Props {
  recipe: RecipeScore
  userId: number
  onCooked?: () => void
  onSkipped?: () => void
}

// ── star rating input ──────────────────────────────────────────────────────
// Renders 5 clickable stars. Hover previews, click confirms.
// This generates event_type=rate which feeds SVD training.
// The user needs MIN_RATINGS_FOR_CF (5) ratings before SVD activates.
function StarRating({
  onRate,
  submitting,
}: {
  onRate: (stars: number) => void
  submitting: boolean
}) {
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

// ── main card ──────────────────────────────────────────────────────────────
export function RecipeCard({ recipe, userId, onCooked, onSkipped }: Props) {
  const [expanded,        setExpanded]        = useState(false)
  const [showIngredients, setShowIngredients] = useState(false)
  const [phase,           setPhase]           = useState<'idle' | 'cooked' | 'rated' | 'skipped'>('idle')
  const [submitting,  setSubmitting]  = useState(false)
  const [listStatus,  setListStatus]  = useState<'idle' | 'adding' | 'added' | 'duplicate'>('idle')

  // Step 1: user clicks "Cook this" — log the cook event, show rating prompt
  const handleCook = async () => {
    setSubmitting(true)
    try {
      await logEvent({
        user_id:    userId,
        recipe_id:  recipe.recipe_id,
        event_type: 'cook',
        n_missing:  recipe.missing_ingredients.length,
      })
      setPhase('cooked')   // reveal star rating
    } finally {
      setSubmitting(false)
    }
  }

  // Step 2: user selects stars (or skips rating) — log rate event, done
  const handleRate = async (stars: number) => {
    if (stars > 0) {
      setSubmitting(true)
      try {
        await logEvent({
          user_id:    userId,
          recipe_id:  recipe.recipe_id,
          event_type: 'rate',
          rating:     stars,
          n_missing:  recipe.missing_ingredients.length,
        })
      } finally {
        setSubmitting(false)
      }
    }
    setPhase('rated')
    // Notify parent after a short delay so user sees confirmation
    setTimeout(() => onCooked?.(), 800)
  }

  const handleSkip = async () => {
    setPhase('skipped')
    await logEvent({
      user_id:    userId,
      recipe_id:  recipe.recipe_id,
      event_type: 'skip',
    })
    onSkipped?.()
  }

  const handleAddToList = async () => {
    if (listStatus !== 'idle') return
    setListStatus('adding')
    const result = await addToShoppingList(userId, {
      ingredients: recipe.missing_ingredients,
      recipe_id:   recipe.recipe_id,
      recipe_name: recipe.recipe_name,
    })
    setListStatus(result.added.length > 0 ? 'added' : 'duplicate')
    setTimeout(() => setListStatus('idle'), 2500)
  }

  // Rated state: green confirmation
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
            Cooked & rated!
          </p>
          <p style={{ fontSize: 12, color: 'var(--green-600)', marginTop: 2 }}>
            Your rating helps improve recommendations.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 4, lineHeight: 1.3 }}>
            <Link
              to={`/recipe/${recipe.recipe_id}`}
              style={{
                color: 'var(--gray-900)', textDecoration: 'none',
                overflow: 'hidden', display: '-webkit-box',
                WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
              }}
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--blue-600)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--gray-900)')}
            >
              {recipe.recipe_name}
            </Link>
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {recipe.minutes && (
              <span className="badge badge-gray">⏱ {recipe.minutes}m</span>
            )}
            {recipe.avg_rating && (
              <span className="badge badge-amber">★ {recipe.avg_rating.toFixed(1)}</span>
            )}
            {recipe.tags.slice(0, 4).map(t => (
              <span key={t} className="badge badge-green">{t}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Missing ingredients */}
      {recipe.missing_ingredients.length > 0 ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
          <p style={{ fontSize: 12, color: 'var(--gray-500)', margin: 0 }}>
            Need: {recipe.missing_ingredients.slice(0, 3).join(', ')}
            {recipe.missing_ingredients.length > 3 && ` +${recipe.missing_ingredients.length - 3} more`}
          </p>
          {phase === 'idle' && (
            <button
              onClick={handleAddToList}
              disabled={listStatus === 'adding'}
              style={{
                fontSize: 11, padding: '2px 7px',
                background: listStatus === 'added' ? 'var(--green-50)'
                          : listStatus === 'duplicate' ? 'var(--gray-50)' : 'none',
                border: `1px solid ${
                  listStatus === 'added' ? 'var(--green-200)'
                  : listStatus === 'duplicate' ? 'var(--gray-200)' : 'var(--gray-200)'}`,
                borderRadius: 4, cursor: listStatus === 'idle' ? 'pointer' : 'default',
                color: listStatus === 'added' ? 'var(--green-700)'
                     : listStatus === 'duplicate' ? 'var(--gray-400)' : 'var(--gray-500)',
                whiteSpace: 'nowrap', lineHeight: '18px',
              }}
            >
              {listStatus === 'added'     ? '✓ Added to list'
               : listStatus === 'duplicate' ? 'Already in list'
               : listStatus === 'adding'    ? '…'
               : '＋ Buy missing'}
            </button>
          )}
        </div>
      ) : (
        <p style={{ fontSize: 12, color: 'var(--green-600)', fontWeight: 500, marginBottom: 10 }}>
          ✓ You have everything!
        </p>
      )}

      {/* Expanders: ingredients + personal fit */}
      <div style={{ display: 'flex', gap: 16 }}>
        <button
          onClick={() => setShowIngredients(s => !s)}
          style={{
            fontSize: 12, color: 'var(--blue-600)', background: 'none', border: 'none',
            textAlign: 'left', padding: 0,
          }}
        >
          {showIngredients ? '▲ Hide ingredients' : '▼ Ingredients'}
        </button>
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            fontSize: 12, color: 'var(--blue-600)', background: 'none', border: 'none',
            textAlign: 'left', padding: 0,
          }}
        >
          {expanded ? '▲ Hide personal fit' : '▼ Personal fit'}
        </button>
      </div>

      {showIngredients && (
        <p style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.6, marginTop: 8 }}>
          {[...recipe.matched_ingredients, ...recipe.missing_ingredients].join(', ')}
        </p>
      )}
      {expanded && <ScoreExplainer recipe={recipe} />}

      {/* Rating prompt — shown after Cook is clicked */}
      {phase === 'cooked' && (
        <div style={{
          marginTop: 12, padding: '12px 0',
          borderTop: '1px solid var(--gray-100)',
          borderBottom: '1px solid var(--gray-100)',
        }}>
          <StarRating onRate={handleRate} submitting={submitting} />
        </div>
      )}

      {/* Action buttons — hidden after cooking */}
      {phase === 'idle' && (
        <div style={{ display: 'flex', gap: 8, marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--gray-100)' }}>
          <button
            className="btn btn-primary"
            style={{ flex: 1 }}
            onClick={handleCook}
            disabled={submitting}
          >
            {submitting ? '…' : '✓ Cook this'}
          </button>
          <button className="btn btn-ghost" onClick={handleSkip}>
            Skip
          </button>
        </div>
      )}

    </div>
  )
}
