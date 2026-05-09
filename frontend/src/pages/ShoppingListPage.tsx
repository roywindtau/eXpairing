// ShoppingListPage.tsx
// Shows the user's buy list. Items can be checked off while shopping
// and removed when done. Grouped by source recipe for context.

import { useEffect, useState } from 'react'
import {
  getShoppingList, removeShoppingItem, toggleShoppingItem,
  clearShoppingList,
} from '../api/client'
import type { ShoppingItem } from '../api/client'

interface Props { userId: number }

export function ShoppingListPage({ userId }: Props) {
  const [items,   setItems]   = useState<ShoppingItem[]>([])
  const [loading, setLoading] = useState(true)

  const load = () =>
    getShoppingList(userId).then(data => {
      setItems(data)
      setLoading(false)
    })

  useEffect(() => { load() }, [userId])

  const handleToggle = async (item: ShoppingItem) => {
    const updated = await toggleShoppingItem(userId, item.id, !item.is_checked)
    setItems(prev => prev.map(i => i.id === item.id ? updated : i))
  }

  const handleRemove = async (itemId: number) => {
    await removeShoppingItem(userId, itemId)
    setItems(prev => prev.filter(i => i.id !== itemId))
  }

  const handleClearChecked = async () => {
    await clearShoppingList(userId, true)
    setItems(prev => prev.filter(i => !i.is_checked))
  }

  if (loading) return <div className="spinner-wrap"><div className="spinner" /></div>

  const checkedCount = items.filter(i => i.is_checked).length

  return (
    <div className="page">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <h1 className="page-title" style={{ margin: 0 }}>Shopping List</h1>
        {checkedCount > 0 && (
          <button className="btn btn-ghost" style={{ fontSize: 13 }} onClick={handleClearChecked}>
            Clear purchased ({checkedCount})
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="empty">
          <div className="empty-icon">🛒</div>
          <h3>Nothing to buy</h3>
          <p>Add missing ingredients from your recipe feed or a recipe detail page.</p>
        </div>
      ) : (
        <>
          <p style={{ fontSize: 13, color: 'var(--gray-400)', marginBottom: 14 }}>
            {items.length} item{items.length !== 1 ? 's' : ''} · {checkedCount} purchased
          </p>
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            {items.map((item, i) => (
              <div
                key={item.id}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '12px 16px',
                  borderBottom: i < items.length - 1 ? '1px solid var(--gray-100)' : 'none',
                  background: item.is_checked ? 'var(--gray-50)' : 'transparent',
                  transition: 'background .15s',
                }}
              >
                {/* Checkbox */}
                <input
                  type="checkbox"
                  checked={item.is_checked}
                  onChange={() => handleToggle(item)}
                  style={{ width: 18, height: 18, cursor: 'pointer', accentColor: 'var(--green-500)', flexShrink: 0 }}
                />

                {/* Name + source */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <span style={{
                    fontWeight: 500,
                    textTransform: 'capitalize',
                    textDecoration: item.is_checked ? 'line-through' : 'none',
                    color: item.is_checked ? 'var(--gray-400)' : 'var(--gray-900)',
                  }}>
                    {item.ingredient}
                  </span>
                  {item.source_recipe_name && (
                    <span style={{ fontSize: 12, color: 'var(--gray-400)', marginLeft: 8 }}>
                      for {item.source_recipe_name}
                    </span>
                  )}
                </div>

                {/* Remove */}
                <button
                  onClick={() => handleRemove(item.id)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: 16, color: 'var(--gray-300)', padding: '2px 6px',
                    lineHeight: 1, flexShrink: 0,
                  }}
                  onMouseEnter={e => (e.currentTarget.style.color = 'var(--red-500)')}
                  onMouseLeave={e => (e.currentTarget.style.color = 'var(--gray-300)')}
                  title="Remove"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          {checkedCount > 0 && (
            <button
              className="btn btn-ghost"
              style={{ marginTop: 16, fontSize: 13, width: '100%' }}
              onClick={handleClearChecked}
            >
              Clear {checkedCount} purchased item{checkedCount !== 1 ? 's' : ''}
            </button>
          )}
        </>
      )}
    </div>
  )
}
