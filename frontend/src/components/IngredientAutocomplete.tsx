import { useState, useEffect, useRef } from 'react'
import api from '../api/client'

interface Props {
  value: string
  onChange: (v: string) => void
  onEnter?: () => void
  placeholder?: string
}

export function IngredientAutocomplete({ value, onChange, onEnter, placeholder }: Props) {
  const [suggestions, setSuggestions]   = useState<string[]>([])
  const [open,        setOpen]          = useState(false)
  const [activeIndex, setActiveIndex]   = useState(-1)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrapRef     = useRef<HTMLDivElement>(null)
  const focusedRef  = useRef(false)  // sync ref — safe to read inside async callbacks

  useEffect(() => {
    if (value.length < 2) { setSuggestions([]); setOpen(false); return }
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      try {
        const { data } = await api.get<string[]>('/pantry/suggest', {
          params: { q: value, limit: 8 },
        })
        setSuggestions(data)
        // Only open if the input is still focused when the response arrives
        if (focusedRef.current) {
          setOpen(data.length > 0)
          setActiveIndex(-1)
        }
      } catch {
        setSuggestions([])
      }
    }, 180)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [value])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const pick = (s: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    focusedRef.current = false  // prevent debounce callback from reopening the dropdown
    onChange(s)
    setSuggestions([])
    setOpen(false)
    setActiveIndex(-1)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (open && suggestions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveIndex(i => Math.min(i + 1, suggestions.length - 1))
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveIndex(i => Math.max(i - 1, 0))
        return
      }
      if (e.key === 'Tab' && activeIndex >= 0) {
        e.preventDefault()
        pick(suggestions[activeIndex])
        return
      }
      if (e.key === 'Enter') {
        if (activeIndex >= 0) {
          e.preventDefault()
          pick(suggestions[activeIndex])
          return
        }
        // No item highlighted — close and let the caller handle Enter
        setOpen(false)
      }
      if (e.key === 'Escape') {
        setOpen(false)
        setActiveIndex(-1)
        return
      }
    }
    if (e.key === 'Enter') onEnter?.()
  }

  return (
    <div ref={wrapRef} style={{ position: 'relative' }}>
      <input
        className="form-input"
        placeholder={placeholder}
        value={value}
        autoComplete="off"
        style={{ width: '100%' }}
        onChange={e => { onChange(e.target.value); setActiveIndex(-1) }}
        onKeyDown={handleKeyDown}
        onFocus={() => { focusedRef.current = true; if (suggestions.length > 0) setOpen(true) }}
        onBlur={() => { focusedRef.current = false; setTimeout(() => setOpen(false), 150) }}
        aria-autocomplete="list"
        aria-expanded={open}
      />
      {open && (
        <ul
          role="listbox"
          data-testid="ingredient-suggestions"
          style={{
            position: 'absolute', top: 'calc(100% + 2px)', left: 0, right: 0,
            zIndex: 200, margin: 0, padding: 0, listStyle: 'none',
            background: 'white',
            border: '1px solid var(--gray-200)',
            borderRadius: 4,
            boxShadow: '0 4px 12px rgba(0,0,0,.08)',
            maxHeight: 220, overflowY: 'auto',
          }}
        >
          {suggestions.map((s, i) => (
            <li
              key={s}
              role="option"
              aria-selected={i === activeIndex}
              onMouseDown={() => pick(s)}
              style={{
                padding: '7px 12px',
                fontSize: 13,
                cursor: 'pointer',
                background: i === activeIndex ? 'var(--blue-50)' : 'transparent',
                color: i === activeIndex ? 'var(--blue-600)' : 'var(--gray-800)',
                borderBottom: i < suggestions.length - 1 ? '1px solid var(--gray-100)' : 'none',
              }}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
