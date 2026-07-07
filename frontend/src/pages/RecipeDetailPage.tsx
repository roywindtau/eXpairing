import { useEffect, useState } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { getRecipeDetail, logEvent } from '../api/client'
import type { RecipeDetail } from '../api/client'
import { WinePairing } from '../components/WinePairing'
import { StarRating } from '../components/StarRating'

interface Props { userId: number }

interface CookNavState {
  fromCook?: boolean
  n_missing?: number
}

// Food.com text is all-lowercase; capitalize the first letter of a sentence.
const sentenceCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

export function RecipeDetailPage({ userId }: Props) {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const cookState = (location.state as CookNavState | null) ?? {}

  const [recipe, setRecipe] = useState<RecipeDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [rated, setRated] = useState(false)
  const [ratedStars, setRatedStars] = useState(0)

  useEffect(() => {
    if (!id) return
    getRecipeDetail(Number(id))
      .then(setRecipe)
      .catch(() => setError('Recipe not found.'))
      .finally(() => setLoading(false))
  }, [id])

  const handleRate = async (stars: number) => {
    if (!recipe) return
    if (stars > 0) {
      setSubmitting(true)
      try {
        await logEvent({
          user_id:    userId,
          recipe_id:  recipe.id,
          event_type: 'rate',
          rating:     stars,
          n_missing:  cookState.n_missing ?? 0,
        })
      } finally {
        setSubmitting(false)
      }
    }
    setRatedStars(stars)
    setRated(true)
  }

  if (loading) return (
    <div className="page">
      <div className="spinner-wrap"><div className="spinner" /></div>
    </div>
  )

  if (error || !recipe) return (
    <div className="page">
      <div className="empty">
        <div className="empty-icon">⚠️</div>
        <h3>Recipe not found</h3>
        <button className="btn btn-ghost" style={{ marginTop: 16 }} onClick={() => navigate(-1)}>
          ← Back
        </button>
      </div>
    </div>
  )

  const showRating = cookState.fromCook && !rated

  return (
    <div className="page" style={{ maxWidth: 690 }}>
      {/* Back */}
      <button
        className="btn btn-ghost"
        style={{ marginBottom: 22, fontSize: 14 }}
        onClick={() => navigate(-1)}
      >
        ← Back
      </button>

      {/* Title + meta */}
      <h1 style={{ fontSize: 34, fontWeight: 600, color: 'var(--gray-900)', marginBottom: 11, lineHeight: 1.2, fontFamily: 'var(--font-display)', letterSpacing: '-.015em', textTransform: 'capitalize' }}>
        {recipe.name}
      </h1>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 9, marginBottom: 22 }}>
        {recipe.minutes && (
          <span className="badge badge-gray" style={{ fontSize: 14, padding: '5px 13px' }}>⏱ {recipe.minutes} min</span>
        )}
        {recipe.n_steps && (
          <span className="badge badge-gray" style={{ fontSize: 14, padding: '5px 13px' }}>📋 {recipe.n_steps} steps</span>
        )}
        {recipe.avg_rating && (
          <span className="badge badge-amber" style={{ fontSize: 14, padding: '5px 13px' }}>
            ★ {recipe.avg_rating.toFixed(1)}
            {recipe.n_ratings > 0 && <span style={{ fontWeight: 400, opacity: .8 }}> ({recipe.n_ratings.toLocaleString()})</span>}
          </span>
        )}
        {recipe.tags.map(t => (
          <span key={t} className="badge badge-green" style={{ fontSize: 14, padding: '5px 13px', textTransform: 'capitalize' }}>{t}</span>
        ))}
      </div>

      {/* Rating — shown after "Cook this" from the feed */}
      {showRating && (
        <section
          className="card"
          style={{
            marginBottom: 28, padding: '20px 24px',
            background: 'var(--green-50)', border: '1px solid var(--green-100)',
          }}
        >
          <StarRating onRate={handleRate} submitting={submitting} />
        </section>
      )}

      {rated && cookState.fromCook && (
        <section
          className="card"
          style={{
            marginBottom: 28, padding: '20px 24px', textAlign: 'center',
            background: 'var(--green-50)', border: '1px solid var(--green-100)',
          }}
        >
          <div style={{ fontSize: 28, marginBottom: 6 }}>✓</div>
          <p style={{ fontSize: 14, color: 'var(--green-700)', fontWeight: 500 }}>
            {ratedStars > 0 ? 'Cooked & rated!' : 'Cooked!'}
          </p>
          {ratedStars > 0 && (
            <p style={{ fontSize: 12, color: 'var(--green-600)', marginTop: 2 }}>
              Your rating helps improve recommendations.
            </p>
          )}
        </section>
      )}

      {/* Description */}
      {recipe.description && (
        <p style={{ fontSize: 16, color: 'var(--gray-600)', lineHeight: 1.7, marginBottom: 32 }}>
          {sentenceCase(recipe.description)}
        </p>
      )}

      {/* Ingredients */}
      <section style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: 22, fontWeight: 600, color: 'var(--gray-800)', marginBottom: 14, fontFamily: 'var(--font-display)' }}>
          Ingredients
        </h2>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 9 }}>
          {recipe.ingredients.map((ing, i) => (
            <li key={i} style={{
              display: 'flex', alignItems: 'center', gap: 11,
              fontSize: 15, color: 'var(--gray-700)', textTransform: 'capitalize',
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: '50%',
                background: 'var(--green-500)', flexShrink: 0,
              }} />
              {ing}
            </li>
          ))}
        </ul>
      </section>

      {/* Steps */}
      {recipe.steps.length > 0 && (
        <section>
          <h2 style={{ fontSize: 22, fontWeight: 600, color: 'var(--gray-800)', marginBottom: 14, fontFamily: 'var(--font-display)' }}>
            Instructions
          </h2>
          <ol style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 18 }}>
            {recipe.steps.map((step, i) => (
              <li key={i} style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                <span style={{
                  minWidth: 32, height: 32, borderRadius: '50%',
                  background: 'linear-gradient(135deg, #46a54b, var(--green-600))', color: 'white',
                  boxShadow: '0 2px 6px rgba(46,125,50,.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 14, fontWeight: 700, flexShrink: 0, marginTop: 1,
                }}>
                  {i + 1}
                </span>
                <p style={{ fontSize: 15, color: 'var(--gray-700)', lineHeight: 1.7, margin: 0 }}>
                  {sentenceCase(step)}
                </p>
              </li>
            ))}
          </ol>
        </section>
      )}

      {recipe.steps.length === 0 && (
        <div className="empty" style={{ marginTop: 16 }}>
          <p style={{ fontSize: 14, color: 'var(--gray-400)' }}>
            No instructions available for this recipe.
          </p>
        </div>
      )}

      {/* Wine pairing */}
      <div style={{ marginTop: 28 }}>
        <WinePairing recipeId={recipe.id} />
      </div>
    </div>
  )
}
