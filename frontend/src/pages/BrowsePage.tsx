// BrowsePage.tsx
// Browse and search the full recipe corpus.
// Not personalized — purely query/tag based.
// Lets the grader/demo audience explore the dataset.

import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import api from '../api/client'

interface RecipeSummary {
  id:          number
  name:        string
  ingredients: string[]
  tags:        string[]
  minutes:     number | null
  avg_rating:  number | null
  n_ratings:   number
}

const POPULAR_TAGS = [
  'vegetarian', 'vegan', 'quick', 'breakfast', 'dessert',
  'italian', 'asian', 'low-carb', 'gluten-free', 'healthy',
]

interface Props { userId: number }

export function BrowsePage({ userId: _userId }: Props) {
  const [query,    setQuery]    = useState('')
  const [tag,      setTag]      = useState('')
  const [results,  setResults]  = useState<RecipeSummary[]>([])
  const [loading,  setLoading]  = useState(false)
  const [searched, setSearched] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const doSearch = async (q: string, t: string) => {
    setLoading(true)
    setSearched(true)
    try {
      const { data } = await api.get<RecipeSummary[]>('/recipes/search', {
        params: { q: q.trim(), tag: t, limit: 40 },
      })
      setResults(data)
    } finally {
      setLoading(false)
    }
  }

  // Debounced search as user types
  useEffect(() => {
    if (!query && !tag) return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(query, tag), 350)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [query, tag])

  // Initial load: top-rated recipes
  useEffect(() => { doSearch('', '') }, [])

  const handleTagClick = (t: string) => {
    setTag(prev => prev === t ? '' : t)
  }

  return (
    <div className="page">
      <h1 className="page-title">Browse recipes</h1>

      {/* Search bar */}
      <div style={{ position: 'relative', marginBottom: 14 }}>
        <span style={{
          position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
          color: 'var(--gray-400)', pointerEvents: 'none',
        }}>🔍</span>
        <input
          className="form-input"
          placeholder="Search by name or ingredient…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          style={{ width: '100%', paddingLeft: 36 }}
        />
      </div>

      {/* Tag filter pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
        {POPULAR_TAGS.map(t => (
          <button
            key={t}
            onClick={() => handleTagClick(t)}
            className={`badge ${tag === t ? 'badge-green' : 'badge-gray'}`}
            style={{
              cursor: 'pointer',
              border: tag === t ? '1px solid var(--green-500)' : '1px solid var(--gray-200)',
              padding: '5px 12px', transition: 'all .15s',
            }}
          >
            {t}
          </button>
        ))}
        {tag && (
          <button
            onClick={() => setTag('')}
            className="badge badge-gray"
            style={{ cursor: 'pointer', border: '1px solid var(--gray-200)', padding: '5px 12px' }}
          >
            ✕ clear filter
          </button>
        )}
      </div>

      {/* Results */}
      {loading && (
        <div className="spinner-wrap"><div className="spinner" /></div>
      )}

      {!loading && searched && results.length === 0 && (
        <div className="empty">
          <div className="empty-icon">🔍</div>
          <h3>No recipes found</h3>
          <p>Try a different search term or tag.</p>
        </div>
      )}

      {!loading && results.length > 0 && (
        <>
          <p style={{ fontSize: 13, color: 'var(--gray-400)', marginBottom: 14 }}>
            {results.length} recipes{query ? ` matching "${query}"` : ''}{tag ? ` tagged ${tag}` : ''}
          </p>
          <div className="recipe-grid">
            {results.map(r => (
              <BrowseCard key={r.id} recipe={r} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ── browse card ─────────────────────────────────────────────────────────────
// Simpler than RecipeCard — no scoring, no actions, just info.

function BrowseCard({ recipe }: { recipe: RecipeSummary }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <h3 style={{ fontSize: 14, fontWeight: 600, lineHeight: 1.3 }}>
        <Link
          to={`/recipe/${recipe.id}`}
          style={{ color: 'var(--gray-900)', textDecoration: 'none' }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--blue-600)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--gray-900)')}
        >
          {recipe.name}
        </Link>
      </h3>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {recipe.minutes && (
          <span className="badge badge-gray">⏱ {recipe.minutes}m</span>
        )}
        {recipe.avg_rating && (
          <span className="badge badge-amber">
            ★ {recipe.avg_rating.toFixed(1)}
            <span style={{ fontWeight: 400, opacity: .7 }}> ({recipe.n_ratings})</span>
          </span>
        )}
        {recipe.tags.map(t => (
          <span key={t} className="badge badge-green">{t}</span>
        ))}
      </div>

      <button
        onClick={() => setExpanded(e => !e)}
        style={{ fontSize: 12, color: 'var(--blue-600)', background: 'none', border: 'none', textAlign: 'left', padding: 0 }}
      >
        {expanded ? '▲ Hide ingredients' : '▼ Show ingredients'}
      </button>

      {expanded && (
        <p style={{ fontSize: 12, color: 'var(--gray-600)', lineHeight: 1.6 }}>
          {recipe.ingredients.join(', ')}
          {recipe.ingredients.length >= 8 && '…'}
        </p>
      )}
    </div>
  )
}
