// VisionScanner.tsx
// Handles the full scan flow:
//   1. User taps "Scan fridge" → file input opens (or drag/drop)
//   2. Image uploaded to POST /vision/scan (or /vision/mock in demo mode)
//   3. Detected items shown in a confirmation table
//   4. User fills in missing expiry dates (inline date inputs)
//   5. User taps "Add X items" → POST /vision/confirm/{userId}
//   6. Parent refreshes pantry list

import { useState, useRef } from 'react'
import type { ChangeEvent } from 'react'
import api from '../api/client'

interface ScannedItem {
  ingredient:  string
  expiry_date: string | null
  raw_name:    string
  quantity:    string | null
}

interface Props {
  userId:     number
  onConfirmed: () => void   // called after successful confirm
  demoMode?:  boolean       // use /vision/mock instead of real scan
}

type Phase = 'idle' | 'scanning' | 'review' | 'confirming' | 'done' | 'error'

export function VisionScanner({ userId, onConfirmed, demoMode = false }: Props) {
  const [phase,   setPhase]   = useState<Phase>('idle')
  const [items,   setItems]   = useState<ScannedItem[]>([])
  const [error,   setError]   = useState('')
  const [preview, setPreview] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // ── step 1: user picks a file ──────────────────────────────────────────
  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setPreview(URL.createObjectURL(file))
    setPhase('scanning')
    setError('')

    try {
      let scanned: ScannedItem[]

      if (demoMode) {
        // Demo mode: call mock endpoint (no API key needed)
        const { data } = await api.get<ScannedItem[]>('/vision/mock')
        scanned = data
      } else {
        // Real scan: upload the image
        const form = new FormData()
        form.append('photo', file)
        const { data } = await api.post<ScannedItem[]>('/vision/scan', form, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        scanned = data
      }

      if (scanned.length === 0) {
        setError('No food products detected. Try a clearer photo.')
        setPhase('error')
        return
      }

      setItems(scanned)
      setPhase('review')
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail ?? 'Scan failed. Check the server connection.'
      setError(msg)
      setPhase('error')
    }
  }

  // ── step 2: user edits expiry dates ───────────────────────────────────
  const updateExpiry = (idx: number, date: string) => {
    setItems(prev => prev.map((item, i) => i === idx ? { ...item, expiry_date: date } : item))
  }

  const updateIngredient = (idx: number, val: string) => {
    setItems(prev => prev.map((item, i) => i === idx ? { ...item, ingredient: val } : item))
  }

  const removeItem = (idx: number) => {
    setItems(prev => prev.filter((_, i) => i !== idx))
  }

  const allHaveExpiry = items.every(i => i.expiry_date)
  const missingCount  = items.filter(i => !i.expiry_date).length

  // ── step 3: confirm and add to pantry ─────────────────────────────────
  const handleConfirm = async () => {
    if (!allHaveExpiry) return
    setPhase('confirming')
    try {
      await api.post(`/vision/confirm/${userId}`, { items })
      setPhase('done')
      setTimeout(() => {
        onConfirmed()
        setPhase('idle')
        setItems([])
        setPreview(null)
      }, 1200)
    } catch {
      setError('Failed to save items. Try again.')
      setPhase('review')
    }
  }

  const reset = () => {
    setPhase('idle')
    setItems([])
    setError('')
    setPreview(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  // ── render ─────────────────────────────────────────────────────────────

  if (phase === 'idle') {
    return (
      <div>
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          style={{ display: 'none' }}
          onChange={handleFile}
        />
        <button
          className="btn btn-ghost"
          onClick={() => fileRef.current?.click()}
          style={{ gap: 8 }}
        >
          <span style={{ fontSize: 16 }}>📷</span>
          {demoMode ? 'Demo scan' : 'Scan fridge'}
        </button>
      </div>
    )
  }

  if (phase === 'scanning') {
    return (
      <div className="card" style={{ textAlign: 'center', padding: 32 }}>
        {preview && (
          <img src={preview} alt="Fridge" style={{
            width: 120, height: 90, objectFit: 'cover',
            borderRadius: 8, marginBottom: 16,
          }} />
        )}
        <div className="spinner" style={{ margin: '0 auto 12px' }} />
        <p style={{ fontSize: 14, color: 'var(--gray-600)' }}>
          {demoMode ? 'Loading demo items…' : 'Scanning with GPT-4o vision…'}
        </p>
      </div>
    )
  }

  if (phase === 'error') {
    return (
      <div className="card" style={{ background: 'var(--red-50)', border: '1px solid #fecaca' }}>
        <p style={{ fontSize: 14, color: 'var(--red-600)', marginBottom: 12 }}>⚠ {error}</p>
        <button className="btn btn-ghost" onClick={reset}>Try again</button>
      </div>
    )
  }

  if (phase === 'done') {
    return (
      <div className="card" style={{ background: 'var(--green-50)', textAlign: 'center', padding: 28 }}>
        <div style={{ fontSize: 32, marginBottom: 8 }}>✓</div>
        <p style={{ fontSize: 15, color: 'var(--green-700)', fontWeight: 500 }}>
          {items.length} items added to your pantry!
        </p>
      </div>
    )
  }

  // review phase
  return (
    <div className="card">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 600 }}>
            Found {items.length} items
          </h3>
          {missingCount > 0 && (
            <p style={{ fontSize: 12, color: 'var(--amber-600)', marginTop: 2 }}>
              ⚠ Fill in {missingCount} missing expiry date{missingCount > 1 ? 's' : ''} to continue
            </p>
          )}
        </div>
        {preview && (
          <img src={preview} alt="Scanned" style={{
            width: 60, height: 45, objectFit: 'cover', borderRadius: 6,
          }} />
        )}
      </div>

      {/* Items table */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
        {items.map((item, idx) => (
          <div key={idx} style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr auto',
            gap: 8, alignItems: 'center',
            padding: '8px 12px',
            background: !item.expiry_date ? 'var(--amber-50)' : 'var(--gray-50)',
            borderRadius: 8,
            border: !item.expiry_date ? '1px solid #fde68a' : '1px solid var(--gray-100)',
          }}>
            <div>
              <input
                className="form-input"
                value={item.ingredient}
                onChange={e => updateIngredient(idx, e.target.value)}
                style={{ fontSize: 13, padding: '5px 8px', width: '100%' }}
              />
              <span style={{ fontSize: 11, color: 'var(--gray-400)', marginTop: 2, display: 'block' }}>
                {item.raw_name}
                {item.quantity && ` · ${item.quantity}`}
              </span>
            </div>
            <input
              type="date"
              className="form-input"
              value={item.expiry_date ?? ''}
              onChange={e => updateExpiry(idx, e.target.value)}
              style={{
                fontSize: 13, padding: '5px 8px',
                borderColor: !item.expiry_date ? 'var(--amber-400)' : undefined,
              }}
            />
            <button
              onClick={() => removeItem(idx)}
              style={{ color: 'var(--gray-400)', fontSize: 16, lineHeight: 1 }}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          className="btn btn-primary"
          onClick={handleConfirm}
          disabled={!allHaveExpiry || phase === 'confirming'}
          style={{ flex: 1, justifyContent: 'center' }}
        >
          {phase === 'confirming' ? 'Adding…' : `Add ${items.length} items to pantry`}
        </button>
        <button className="btn btn-ghost" onClick={reset}>
          Cancel
        </button>
      </div>
    </div>
  )
}
