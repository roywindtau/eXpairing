// ProfilePage.tsx
// Shows the user's current profile and lets them update beta + diet tags.
// Changes take effect on next recipe feed load.

import { useEffect, useState } from 'react'
import { getUser, updateUser } from '../api/client'
import api from '../api/client'
import type { UserProfile } from '../api/client'

interface Props { userId: number }

const DIET_OPTIONS = [
  'vegetarian', 'vegan', 'gluten-free', 'dairy-free',
  'keto', 'paleo', 'low-carb', 'nut-free',
]

export function ProfilePage({ userId }: Props) {
  const [profile,  setProfile]  = useState<UserProfile | null>(null)
  const [beta,     setBeta]      = useState(0.35)
  const [dietTags, setDietTags]  = useState<string[]>([])
  const [saving,   setSaving]    = useState(false)
  const [saved,    setSaved]     = useState(false)
  const [stats,    setStats]     = useState<{
    n_ratings: number
    warm_cf_progress_pct: number
    is_warm: boolean
    n_cooked: number
    ratings_for_warm_cf: number
    beta: number
    revealed_beta: number | null
    avg_missing: number | null
  } | null>(null)

  useEffect(() => {
    getUser(userId).then(u => {
      setProfile(u)
      setBeta(u.beta)
      setDietTags(u.diet_tags ? u.diet_tags.split(',').map(t => t.trim()) : [])
    })
    api.get(`/users/${userId}/stats`).then(r => setStats(r.data))
  }, [userId])

  const toggleTag = (tag: string) =>
    setDietTags(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag])

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateUser(userId, {
        name:      profile?.name ?? undefined,
        beta,
        diet_tags: dietTags.length > 0 ? dietTags.join(',') : undefined,
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const betaLabel = beta < 0.25 ? 'Happy to buy a few extra ingredients'
    : beta < 0.55 ? 'Some flexibility is fine'
    : beta < 0.8  ? 'Prefer to use what I have'
    : 'Only suggest recipes I can cook right now'

  if (!profile) return <div className="spinner-wrap"><div className="spinner" /></div>

  return (
    <div className="page" style={{ maxWidth: 720 }}>
      <h1 className="page-title" style={{ fontSize: '2rem' }}>Profile</h1>

      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 34, padding: '2.25rem' }}>
        {/* Beta */}
        <div className="form-group">
          <label className="form-label" style={{ fontSize: 16 }}>Recipe suggestion style</label>
          <input
            type="range" min={0.05} max={0.95} step={0.05}
            value={beta}
            onChange={e => setBeta(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--green-600)' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, color: 'var(--gray-400)' }}>
            <span>Discover new recipes</span>
            <span>Use what I have</span>
          </div>
          <p style={{ fontSize: 15, color: 'var(--green-700)', fontStyle: 'italic', marginTop: 6 }}>
            {betaLabel}
          </p>
          <p style={{ fontSize: 12, color: 'var(--gray-400)', marginTop: 2 }}>
            This also adjusts automatically based on what you actually cook.
          </p>
          {stats?.revealed_beta !== null && stats?.revealed_beta !== undefined &&
           Math.abs((stats.revealed_beta) - beta) > 0.1 && (
            <p style={{ fontSize: 12, color: 'var(--amber-600)', marginTop: 6, fontStyle: 'italic' }}>
              Your cooking history suggests {Math.round(stats.revealed_beta * 100)}% availability focus
              — your slider is set to {Math.round(beta * 100)}%.
              {stats.avg_missing !== null && ` (avg ${stats.avg_missing} missing ingredients per cook)`}
            </p>
          )}
        </div>

        {/* CF status + rating progress */}
        <div style={{ padding: 20, background: 'var(--surface-2)', border: '1px solid var(--gray-200)', borderRadius: 'var(--radius-md)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
            <div>
              <p style={{ fontSize: 15, fontWeight: 600 }}>Personalization status</p>
              <p style={{ fontSize: 13.5, color: 'var(--gray-500)', marginTop: 4 }}>
                {stats?.is_warm
                  ? 'Full CF active — recommendations use your rating history.'
                  : `Cold start mode — rate ${(stats?.ratings_for_warm_cf ?? 5) - (stats?.n_ratings ?? 0)} more recipes to unlock full personalization.`}
              </p>
            </div>
            <span
              className={`badge ${stats?.is_warm ? 'badge-green' : 'badge-blue'}`}
              style={{ flexShrink: 0, whiteSpace: 'nowrap' }}
            >
              {stats?.is_warm ? 'Personalized' : 'Cold start'}
            </span>
          </div>
          {!stats?.is_warm && stats && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--gray-500)', marginBottom: 4 }}>
                <span>Rating progress</span>
                <span>{stats.n_ratings} / {stats.ratings_for_warm_cf} ratings</span>
              </div>
              <div style={{ height: 6, background: 'var(--gray-200)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', borderRadius: 3,
                  width: `${stats.warm_cf_progress_pct}%`,
                  background: 'var(--blue-500)',
                  transition: 'width .4s',
                }} />
              </div>
            </div>
          )}
          {stats && (
            <div style={{ display: 'flex', gap: 20, marginTop: 14 }}>
              <span style={{ fontSize: 14, color: 'var(--gray-500)' }}>🍳 {stats.n_cooked} cooked</span>
              <span style={{ fontSize: 14, color: 'var(--gray-500)' }}>★ {stats.n_ratings} rated</span>
            </div>
          )}
        </div>

        {/* Diet tags */}
        <div className="form-group">
          <label className="form-label" style={{ fontSize: 16 }}>Dietary preferences</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 6 }}>
            {DIET_OPTIONS.map(tag => (
              <button key={tag} onClick={() => toggleTag(tag)}
                className={`badge ${dietTags.includes(tag) ? 'badge-green' : 'badge-gray'}`}
                style={{
                  cursor: 'pointer', fontSize: 14,
                  border: dietTags.includes(tag) ? '1px solid var(--green-500)' : '1px solid var(--gray-200)',
                  padding: '8px 16px', transition: 'all .15s',
                }}>
                {tag}
              </button>
            ))}
          </div>
          <p style={{ fontSize: 13, color: 'var(--gray-400)', marginTop: 8 }}>
            Used to filter and seed cold-start recommendations.
          </p>
        </div>

        <button className="btn btn-primary" onClick={handleSave} disabled={saving}
          style={{ alignSelf: 'flex-start', fontSize: 15, padding: '10px 22px' }}>
          {saved ? '✓ Saved' : saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>
    </div>
  )
}
