// ScoreExplainer.tsx
// Shows the 4-component score breakdown for a recipe.
// This is the "why this recipe" panel shown in the recipe card.

import type { RecipeScore } from '../api/client'

interface Props {
  recipe: RecipeScore
}

interface BarProps {
  label: string
  value: number      // 0..1
  color: string
  hint: string
  unavailable?: boolean   // model not loaded → "not trained"
  noData?: boolean        // model loaded but no coverage for this recipe → "—"
}

function ScoreBar({ label, value, color, hint, unavailable, noData }: BarProps) {
  const pct = Math.round(value * 100)
  const dim = unavailable || noData
  const barColor = dim ? 'var(--gray-200)' : color
  const textColor = dim ? 'var(--gray-400)' : 'var(--gray-600)'
  return (
    <div title={hint} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13 }}>
      <span style={{ width: 148, color: textColor, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: 'var(--gray-100)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: barColor, borderRadius: 3, transition: 'width .4s' }} />
      </div>
      {unavailable
        ? <span style={{ width: 80, textAlign: 'right', fontSize: 11, color: 'var(--gray-400)', fontStyle: 'italic' }}>not trained</span>
        : noData
          ? <span style={{ width: 80, textAlign: 'right', fontSize: 11, color: 'var(--gray-400)', fontStyle: 'italic' }}>no data</span>
          : <span style={{ width: 34, textAlign: 'right', fontSize: 12, color: 'var(--gray-500)' }}>{pct}%</span>
      }
    </div>
  )
}

const strategyLabel: Record<string, string> = {
  biased_mf:             'Matrix factorization',
  blended:               'Blended CF',
  item_based_cold_start: 'Item-based CF',
  none:                  'Not available',
}

export function ScoreExplainer({ recipe }: Props) {
  const cfLabel = strategyLabel[recipe.cf_strategy] ?? recipe.cf_strategy
  const cfBadgeClass = recipe.cf_strategy === 'biased_mf' ? 'badge-green'
    : recipe.cf_strategy === 'item_based_cold_start' ? 'badge-blue'
    : 'badge-gray'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, paddingTop: 12 }}>
      <div style={{ marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '.04em' }}>
          Why this recipe
        </span>
      </div>

      <ScoreBar
        label="Expiry urgency"
        value={recipe.expiry_urgency}
        color="var(--red-400)"
        hint="How urgently your pantry items match this recipe's ingredients"
      />
      <ScoreBar
        label="Ingredient match"
        value={recipe.match_ratio}
        color="var(--green-500)"
        hint={`${recipe.matched_ingredients.length} of ${recipe.total_ingredients} ingredients in your pantry`}
      />
      <ScoreBar
        label="Community score (CF)"
        value={recipe.cf_score}
        color="var(--blue-500)"
        hint="Collaborative filtering — rating patterns from users with similar tastes"
        unavailable={recipe.cf_strategy === 'none'}
        noData={recipe.cf_strategy === 'item_based_cold_start' && recipe.cf_score === 0}
      />
      <ScoreBar
        label="Profile match (CB)"
        value={recipe.cb_score}
        color="var(--amber-400)"
        hint="Content-based — TF-IDF similarity between your pantry ingredients and this recipe"
        unavailable={!recipe.cb_model_available}
      />

      {/* CF strategy badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
        <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>CF mode:</span>
        <span className={`badge ${cfBadgeClass}`} style={{ fontSize: 11 }}>{cfLabel}</span>
      </div>

      {/* Missing ingredients */}
      {recipe.missing_ingredients.length > 0 && (
        <div style={{ marginTop: 4 }}>
          <span style={{ fontSize: 12, color: 'var(--gray-500)' }}>Need to buy: </span>
          <span style={{ fontSize: 12, color: 'var(--gray-700)' }}>
            {recipe.missing_ingredients.join(', ')}
          </span>
        </div>
      )}
    </div>
  )
}
