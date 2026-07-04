// OnboardingPage.tsx
// First-run page. Creates a user with:
//   - Name (optional)
//   - Beta (waste aversion slider)
//   - Diet tags (checkboxes)
// Navigates to /pantry after creation.

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createUser } from '../api/client'

const DIET_OPTIONS = [
  'vegetarian', 'vegan', 'gluten-free', 'dairy-free',
  'keto', 'paleo', 'low-carb', 'nut-free',
]

interface Props {
  onCreated: (userId: number) => void
}

export function OnboardingPage({ onCreated }: Props) {
  const navigate = useNavigate()
  const [name,        setName]        = useState('')
  const [beta,        setBeta]        = useState(0.35)
  const [dietTags,    setDietTags]    = useState<string[]>([])
  const [submitting,  setSubmitting]  = useState(false)
  const [error,       setError]       = useState<string | null>(null)

  const toggleTag = (tag: string) =>
    setDietTags(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag])

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const user = await createUser({
        name: name.trim() || undefined,
        beta,
        diet_tags: dietTags.length > 0 ? dietTags.join(',') : undefined,
      })
      onCreated(user.id)
      navigate('/pantry')
    } catch {
      setError('Could not connect to the server. Make sure the backend is running.')
      setSubmitting(false)
    }
  }

  const betaLabel = beta < 0.25
    ? 'I\'m happy to buy a few extra ingredients'
    : beta < 0.55
    ? 'Some flexibility is fine'
    : beta < 0.8
    ? 'Prefer to use what I have'
    : 'Only suggest recipes I can cook right now'

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
      <div className="card" style={{ maxWidth: 480, width: '100%', padding: '2rem 1.75rem', borderRadius: 20 }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{
            fontSize: 34, width: 72, height: 72, margin: '0 auto 14px',
            borderRadius: 24, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'linear-gradient(135deg, #5cb860, var(--green-600))',
            boxShadow: '0 6px 18px rgba(46,125,50,.3), inset 0 1px 0 rgba(255,255,255,.25)',
          }}>🍳</div>
          <h1 style={{ fontSize: 27, fontWeight: 600, color: 'var(--green-700)', fontFamily: 'var(--font-display)', letterSpacing: '-.015em' }}>eXpairing</h1>
          <p style={{ fontSize: 14, color: 'var(--gray-500)', marginTop: 4 }}>
            Rank recipes to minimize food waste
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Name */}
          <div className="form-group">
            <label className="form-label">Your name (optional)</label>
            <input
              className="form-input"
              placeholder="e.g. Rubi"
              value={name}
              onChange={e => setName(e.target.value)}
            />
          </div>

          {/* Beta slider */}
          <div className="form-group">
            <label className="form-label">Recipe suggestions style</label>
            <input
              type="range" min={0.05} max={0.95} step={0.05}
              value={beta}
              onChange={e => setBeta(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--green-600)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--gray-400)', marginTop: 2 }}>
              <span>Discover new recipes</span>
              <span>Use what I have</span>
            </div>
            <p style={{ fontSize: 13, color: 'var(--green-700)', fontStyle: 'italic', marginTop: 4 }}>
              "{betaLabel}"
            </p>
            <p style={{ fontSize: 12, color: 'var(--gray-400)', marginTop: 2 }}>
              This adjusts automatically as you cook — it's just a starting point.
            </p>
          </div>

          {/* Diet tags */}
          <div className="form-group">
            <label className="form-label">Dietary preferences (optional)</label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 4 }}>
              {DIET_OPTIONS.map(tag => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={`badge ${dietTags.includes(tag) ? 'badge-green' : 'badge-gray'}`}
                  style={{
                    cursor: 'pointer',
                    border: dietTags.includes(tag) ? '1px solid var(--green-500)' : '1px solid var(--gray-200)',
                    padding: '5px 12px',
                    transition: 'all .15s',
                  }}
                >
                  {tag}
                </button>
              ))}
            </div>
            <p style={{ fontSize: 12, color: 'var(--gray-400)', marginTop: 6 }}>
              Used to personalize recommendations from your first visit.
            </p>
          </div>

          {error && (
            <p style={{ fontSize: 13, color: 'var(--red-600)', background: 'var(--red-50)', padding: '8px 12px', borderRadius: 'var(--radius-sm)' }}>
              {error}
            </p>
          )}

          <button
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '10px 0', fontSize: 15 }}
            onClick={handleSubmit}
            disabled={submitting}
          >
            {submitting ? 'Setting up…' : 'Get started →'}
          </button>
        </div>
      </div>
    </div>
  )
}
