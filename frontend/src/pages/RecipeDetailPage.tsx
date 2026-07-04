import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getRecipeDetail } from '../api/client'
import type { RecipeDetail } from '../api/client'
import { WinePairing } from '../components/WinePairing'

// userId is accepted (App passes it to every page route) but pairing is pure
// content-based, so it isn't used here.
interface Props { userId: number }

export function RecipeDetailPage(_: Props) {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [recipe, setRecipe] = useState<RecipeDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    getRecipeDetail(Number(id))
      .then(setRecipe)
      .catch(() => setError('Recipe not found.'))
      .finally(() => setLoading(false))
  }, [id])

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

  return (
    <div className="page" style={{ maxWidth: 680 }}>
      {/* Back */}
      <button
        className="btn btn-ghost"
        style={{ marginBottom: 20, fontSize: 13 }}
        onClick={() => navigate(-1)}
      >
        ← Back
      </button>

      {/* Title + meta */}
      <h1 style={{ fontSize: 27, fontWeight: 600, color: 'var(--gray-900)', marginBottom: 8, lineHeight: 1.25, fontFamily: 'var(--font-display)', letterSpacing: '-.015em' }}>
        {recipe.name}
      </h1>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
        {recipe.minutes && (
          <span className="badge badge-gray" style={{ fontSize: 13 }}>⏱ {recipe.minutes} min</span>
        )}
        {recipe.n_steps && (
          <span className="badge badge-gray" style={{ fontSize: 13 }}>📋 {recipe.n_steps} steps</span>
        )}
        {recipe.avg_rating && (
          <span className="badge badge-amber" style={{ fontSize: 13 }}>
            ★ {recipe.avg_rating.toFixed(1)}
            {recipe.n_ratings > 0 && <span style={{ fontWeight: 400, opacity: .8 }}> ({recipe.n_ratings.toLocaleString()})</span>}
          </span>
        )}
        {recipe.tags.map(t => (
          <span key={t} className="badge badge-green" style={{ fontSize: 13 }}>{t}</span>
        ))}
      </div>

      {/* Description */}
      {recipe.description && (
        <p style={{ fontSize: 15, color: 'var(--gray-600)', lineHeight: 1.6, marginBottom: 28 }}>
          {recipe.description}
        </p>
      )}

      {/* Ingredients */}
      <section style={{ marginBottom: 28 }}>
        <h2 style={{ fontSize: 17, fontWeight: 600, color: 'var(--gray-800)', marginBottom: 12, fontFamily: 'var(--font-display)' }}>
          Ingredients
        </h2>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {recipe.ingredients.map((ing, i) => (
            <li key={i} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              fontSize: 14, color: 'var(--gray-700)',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
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
          <h2 style={{ fontSize: 17, fontWeight: 600, color: 'var(--gray-800)', marginBottom: 12, fontFamily: 'var(--font-display)' }}>
            Instructions
          </h2>
          <ol style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 16 }}>
            {recipe.steps.map((step, i) => (
              <li key={i} style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
                <span style={{
                  minWidth: 28, height: 28, borderRadius: '50%',
                  background: 'linear-gradient(135deg, #46a54b, var(--green-600))', color: 'white',
                  boxShadow: '0 2px 6px rgba(46,125,50,.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, fontWeight: 700, flexShrink: 0, marginTop: 1,
                }}>
                  {i + 1}
                </span>
                <p style={{ fontSize: 14, color: 'var(--gray-700)', lineHeight: 1.6, margin: 0 }}>
                  {step}
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
