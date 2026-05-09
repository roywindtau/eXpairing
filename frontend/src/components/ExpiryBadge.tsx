// ExpiryBadge.tsx
// Shows days until expiry with urgency-coded color.
// Red < 3 days, amber 3-7, green > 7.

interface Props {
  expiryDate: string   // ISO "YYYY-MM-DD"
  showBar?: boolean
}

function daysUntil(iso: string): number {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const exp = new Date(iso)
  exp.setHours(0, 0, 0, 0)
  return Math.round((exp.getTime() - today.getTime()) / 86_400_000)
}

export function ExpiryBadge({ expiryDate, showBar = false }: Props) {
  const days = daysUntil(expiryDate)

  let cls = 'badge-green'
  let label = ''

  if (days < 0) {
    cls = 'badge-red'
    label = 'Expired'
  } else if (days === 0) {
    cls = 'badge-red'
    label = 'Today'
  } else if (days === 1) {
    cls = 'badge-red'
    label = 'Tomorrow'
  } else if (days <= 3) {
    cls = 'badge-red'
    label = `${days}d left`
  } else if (days <= 7) {
    cls = 'badge-amber'
    label = `${days}d left`
  } else {
    cls = 'badge-green'
    label = `${days}d left`
  }

  // Urgency bar: 100% = today, 0% = 14+ days away
  const urgencyPct = Math.max(0, Math.min(100, 100 - (days / 14) * 100))
  const barColor = days <= 3 ? 'var(--red-400)' : days <= 7 ? 'var(--amber-400)' : 'var(--green-500)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span className={`badge ${cls}`}>{label}</span>
      {showBar && (
        <div style={{ height: 3, width: 64, background: 'var(--gray-200)', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${urgencyPct}%`, background: barColor, borderRadius: 2, transition: 'width .3s' }} />
        </div>
      )}
    </div>
  )
}
