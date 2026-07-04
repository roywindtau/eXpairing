// RecipeCard.tsx
// Shows a recipe with match ring, score explainer, and three interaction modes:
//   Cook  -> logs event_type=cook, reveals star rating input
//   Rate  -> logs event_type=rate with 1-5 stars (feeds SVD training)
//   Skip  -> logs event_type=skip, hides card from feed

import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { RecipeScore } from '../api/client'
import { ScoreExplainer } from './ScoreExplainer'
import { ScoreRing } from './ScoreRing'
import { WinePairing } from './WinePairing'
import { logEvent, addToShoppingList } from '../api/client'

interface Props {
  recipe: RecipeScore
  userId: number
  onCooked?: () => void
  onSkipped?: () => void
}

// Card look: a bright warm off-white base with a score-colored left accent
// edge (green → amber → terracotta by match quality), tying each card to its
// score ring.
function cardStyle(scorePct: number): React.CSSProperties {
  const edge = scorePct >= 75 ? 'var(--green-600)' : scorePct >= 55 ? 'var(--green-500)'
    : scorePct >= 35 ? 'var(--amber-400)' : 'var(--red-400)'
  return { background: '#fffdf8', borderColor: '#efe9dc', borderLeft: `6px solid ${edge}` }
}

// Recipe names arrive lower-cased from the dataset ("kentucky hot browns").
// Present them in Title Case, leaving short joiner words lowercase.
const SMALL_WORDS = new Set(['a', 'an', 'and', 'the', 'of', 'or', 'in', 'on', 'with', 'to', 'for', 'ii', 'iii'])
function titleCase(name: string): string {
  // The dataset strips apostrophes, so possessives arrive as a lone "s"
  // ("sizzler s cheese toast"). Re-attach it as "'s" before title-casing.
  const cleaned = name.trim().replace(/\b(\w+) s\b/gi, "$1's")
  const words = cleaned.split(/\s+/)
  return words
    .map((w, i) => {
      const lower = w.toLowerCase()
      if (lower === 'ii')  return 'II'
      if (lower === 'iii') return 'III'
      if (i > 0 && i < words.length - 1 && SMALL_WORDS.has(lower)) return lower
      return w.charAt(0).toUpperCase() + w.slice(1)
    })
    .join(' ')
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

  const matchPct = Math.round(recipe.match_ratio * 100)
  const missingCount = recipe.missing_ingredients.length

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0, padding: '1.75rem 1.75rem 1.5rem', position: 'relative', ...cardStyle(Math.round(recipe.final_score * 100)) }}>

      {/* Header: title + meta on the left, match-score ring on the right */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 18, marginBottom: 18 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 style={{ fontSize: 24, fontWeight: 600, marginBottom: 10, lineHeight: 1.22 }}>
            <Link
              to={`/recipe/${recipe.recipe_id}`}
              style={{
                color: 'var(--gray-900)', textDecoration: 'none',
                overflow: 'hidden', display: '-webkit-box',
                WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
              }}
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--green-700)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--gray-900)')}
            >
              {titleCase(recipe.recipe_name)}
            </Link>
          </h3>

          {/* Quiet meta line: time · rating · pantry match */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: 14.5, color: 'var(--gray-500)', flexWrap: 'wrap' }}>
            {!!recipe.minutes  && <span>⏱ {recipe.minutes}m</span>}
            {!!recipe.avg_rating && <span style={{ color: 'var(--amber-600)' }}>★ {recipe.avg_rating.toFixed(1)}</span>}
            <span style={{ color: missingCount === 0 ? 'var(--green-700)' : 'var(--gray-500)', fontWeight: missingCount === 0 ? 600 : 400 }}>
              {missingCount === 0 ? '✓ Have everything' : `${matchPct}% in pantry`}
            </span>
          </div>
        </div>

        <ScoreRing value={recipe.final_score} size={72} label="match" />
      </div>

      {/* Buy missing — surfaced in the body when ingredients are short */}
      {phase === 'idle' && missingCount > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>
            Need {missingCount} item{missingCount > 1 ? 's' : ''}
          </span>
          <button
            onClick={handleAddToList}
            disabled={listStatus === 'adding'}
            style={{
              fontSize: 11, padding: '3px 10px',
              background: listStatus === 'added' ? 'var(--green-50)'
                        : listStatus === 'duplicate' ? 'var(--gray-50)' : 'none',
              border: `1px solid ${listStatus === 'added' ? 'var(--green-200)' : 'var(--gray-200)'}`,
              borderRadius: 999, cursor: listStatus === 'idle' ? 'pointer' : 'default',
              color: listStatus === 'added' ? 'var(--green-700)'
                   : listStatus === 'duplicate' ? 'var(--gray-400)' : 'var(--gray-600)',
              whiteSpace: 'nowrap', lineHeight: '18px', fontWeight: 500,
            }}
          >
            {listStatus === 'added'      ? '✓ Added to list'
             : listStatus === 'duplicate' ? 'Already in list'
             : listStatus === 'adding'    ? '…'
             : '＋ Buy missing'}
          </button>
        </div>
      )}

      {/* Primary actions */}
      {phase === 'idle' && (
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="btn btn-primary"
            style={{ fontSize: 15, padding: '.6rem 1.3rem' }}
            onClick={handleCook}
            disabled={submitting}
          >
            {submitting ? '…' : '✓ Cook this'}
          </button>
          <button className="btn btn-ghost" style={{ fontSize: 15, padding: '.6rem 1.3rem' }} onClick={handleSkip}>
            Skip
          </button>
        </div>
      )}

      {/* Details — everything secondary lives behind one quiet toggle */}
      {phase === 'idle' && (
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            fontSize: 13, fontWeight: 500, color: 'var(--gray-400)',
            background: 'none', border: 'none', cursor: 'pointer',
            textAlign: 'center', padding: '12px 0 0', marginTop: 6,
          }}
        >
          {expanded ? 'Hide details' : 'Details'}
        </button>
      )}

      {expanded && phase === 'idle' && (
        <div style={{ marginTop: 12, paddingTop: 14, borderTop: '1px solid var(--gray-100)', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Full ingredient list */}
          <button
            onClick={() => setShowIngredients(s => !s)}
            style={{
              fontSize: 12, fontWeight: 500, color: 'var(--green-700)', background: 'none',
              border: 'none', textAlign: 'left', padding: 0, cursor: 'pointer',
            }}
          >
            {showIngredients ? '▲ Hide ingredients' : '▼ Ingredients'}
          </button>
          {showIngredients && (
            <p style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.6, margin: 0 }}>
              {[...recipe.matched_ingredients, ...recipe.missing_ingredients].join(', ')}
            </p>
          )}

          {/* Score breakdown */}
          <ScoreExplainer recipe={recipe} />
        </div>
      )}

      {/* Rating prompt — shown after Cook is clicked */}
      {phase === 'cooked' && (
        <div style={{
          marginTop: 12, padding: '12px 0',
          borderTop: '1px solid var(--gray-100)',
          borderBottom: '1px solid var(--gray-100)',
        }}>
          <StarRating onRate={handleRate} submitting={submitting} />
          <div style={{ marginTop: 12 }}>
            <WinePairing recipeId={recipe.recipe_id} />
          </div>
        </div>
      )}

    </div>
  )
}
