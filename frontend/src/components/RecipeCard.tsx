// RecipeCard.tsx
// Shows a recipe with match ring, score explainer, and two interaction modes:
//   Cook  -> logs event_type=cook, navigates to recipe detail for rating
//   Skip  -> logs event_type=skip, hides card from feed

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { RecipeScore } from '../api/client'
import { ScoreExplainer } from './ScoreExplainer'
import { ScoreRing } from './ScoreRing'
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

export function RecipeCard({ recipe, userId, onCooked, onSkipped }: Props) {
  const navigate = useNavigate()
  const [expanded,        setExpanded]        = useState(false)
  const [showIngredients, setShowIngredients] = useState(false)
  const [skipped,         setSkipped]         = useState(false)
  const [submitting,      setSubmitting]      = useState(false)
  const [listStatus,      setListStatus]      = useState<'idle' | 'adding' | 'added' | 'duplicate'>('idle')

  const handleCook = async () => {
    setSubmitting(true)
    try {
      await logEvent({
        user_id:    userId,
        recipe_id:  recipe.recipe_id,
        event_type: 'cook',
        n_missing:  recipe.missing_ingredients.length,
      })
      onCooked?.()
      navigate(`/recipe/${recipe.recipe_id}`, {
        state: {
          fromCook:  true,
          n_missing: recipe.missing_ingredients.length,
        },
      })
    } finally {
      setSubmitting(false)
    }
  }

  const handleSkip = async () => {
    setSkipped(true)
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

  if (skipped) return null

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
      {missingCount > 0 && (
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

      {/* Details — everything secondary lives behind one quiet toggle */}
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

      {expanded && (
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

    </div>
  )
}
