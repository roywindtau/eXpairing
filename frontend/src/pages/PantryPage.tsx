// PantryPage.tsx
// Shows the user's current pantry sorted by expiry date.
// Allows adding items manually and deleting existing ones.

import { useEffect, useState, useRef } from 'react'
import { getPantry, addPantryItem, deletePantryItem } from '../api/client'
import type { PantryItem } from '../api/client'
import { ExpiryBadge } from '../components/ExpiryBadge'
import { VisionScanner } from '../components/VisionScanner'
import { IngredientAutocomplete } from '../components/IngredientAutocomplete'

interface Props { userId: number }

function daysUntil(iso: string): number {
  const today = new Date(); today.setHours(0,0,0,0)
  const exp   = new Date(iso); exp.setHours(0,0,0,0)
  return Math.round((exp.getTime() - today.getTime()) / 86_400_000)
}

function rowBg(iso: string): string {
  const d = daysUntil(iso)
  if (d < 0) return 'var(--red-50)'
  if (d <= 3) return '#fff5f5'
  if (d <= 7) return 'var(--amber-50)'
  return 'transparent'
}

export function PantryPage({ userId }: Props) {
  const [items,   setItems]   = useState<PantryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [adding,  setAdding]  = useState(false)
  const firstLoad = useRef(true)

  // Add form state
  const [ingredient,  setIngredient]  = useState('')
  const [expiryDate,  setExpiryDate]  = useState('')
  const [quantity,    setQuantity]    = useState('')
  const [saving,      setSaving]      = useState(false)
  const [formError,   setFormError]   = useState('')

  const load = () => {
    if (firstLoad.current) setLoading(true)
    getPantry(userId)
      .then(data => {
        setItems(data)
        if (firstLoad.current) { setLoading(false); firstLoad.current = false }
      })
  }

  useEffect(() => { load() }, [userId])

  const handleAdd = async () => {
    if (!ingredient.trim()) { setFormError('Ingredient name is required'); return }
    if (!expiryDate)         { setFormError('Expiry date is required'); return }
    setFormError('')
    setSaving(true)
    try {
      const item = await addPantryItem(userId, {
        ingredient: ingredient.trim().toLowerCase(),
        expiry_date: expiryDate,
        quantity: quantity.trim() || null,
        raw_name: null,
      })
      setItems(prev => [...prev, item].sort((a, b) => a.expiry_date.localeCompare(b.expiry_date)))
      setIngredient(''); setExpiryDate(''); setQuantity('')
      setAdding(false)
    } catch {
      setFormError('Failed to add item')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (item: PantryItem) => {
    await deletePantryItem(userId, item.id)
    setItems(prev => prev.filter(i => i.id !== item.id))
  }

  if (loading) return <div className="spinner-wrap"><div className="spinner" /></div>

  return (
    <div className="page">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <h1 className="page-title" style={{ margin: 0 }}>My Pantry</h1>
        <div style={{ display: 'flex', gap: 8 }}>
          <VisionScanner userId={userId} onConfirmed={load} demoMode={!import.meta.env.VITE_OPENAI_KEY} />
          <button className="btn btn-primary" onClick={() => setAdding(a => !a)}>
            {adding ? '✕ Cancel' : '+ Add item'}
          </button>
        </div>
      </div>

      {/* Add form */}
      {adding && (
        <div className="card" style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 14 }}>Add pantry item</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
            <div className="form-group">
              <label className="form-label">Ingredient *</label>
              <IngredientAutocomplete
                value={ingredient}
                onChange={setIngredient}
                onEnter={handleAdd}
                placeholder="e.g. milk"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Expiry date *</label>
              <input type="date" className="form-input"
                value={expiryDate} onChange={e => setExpiryDate(e.target.value)} />
            </div>
            <div className="form-group">
              <label className="form-label">Quantity</label>
              <input className="form-input" placeholder="e.g. 500ml"
                value={quantity} onChange={e => setQuantity(e.target.value)} />
            </div>
          </div>
          {formError && <p style={{ fontSize: 13, color: 'var(--red-600)', marginBottom: 10 }}>{formError}</p>}
          <button className="btn btn-primary" onClick={handleAdd} disabled={saving}>
            {saving ? 'Adding…' : 'Add'}
          </button>
        </div>
      )}

      {/* Summary pills */}
      {items.length > 0 && (() => {
        const expiring = items.filter(i => daysUntil(i.expiry_date) <= 3).length
        const expired  = items.filter(i => daysUntil(i.expiry_date) < 0).length
        return (
          <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
            <span className="badge badge-gray">{items.length} items</span>
            {expiring > 0 && <span className="badge badge-red">⚠ {expiring} expiring soon</span>}
            {expired  > 0 && <span className="badge badge-red">✕ {expired} expired</span>}
          </div>
        )
      })()}

      {/* Pantry list */}
      {items.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">🧺</div>
          <h3>Your pantry is empty</h3>
          <p>Add ingredients to start getting recipe recommendations.</p>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {items.map((item, i) => (
            <div key={item.id}
              style={{
                display: 'flex', alignItems: 'center', gap: 14,
                padding: '12px 18px',
                background: rowBg(item.expiry_date),
                borderBottom: i < items.length - 1 ? '1px solid var(--gray-100)' : 'none',
              }}>
              <div style={{ flex: 1 }}>
                <span style={{ fontWeight: 500, textTransform: 'capitalize' }}>{item.ingredient}</span>
                {item.quantity && (
                  <span style={{ fontSize: 13, color: 'var(--gray-400)', marginLeft: 8 }}>{item.quantity}</span>
                )}
              </div>
              <ExpiryBadge expiryDate={item.expiry_date} showBar />
              <button
                className="btn btn-danger"
                style={{ padding: '4px 10px', fontSize: 13 }}
                onClick={() => handleDelete(item)}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
