// ScoreRing.tsx
// A small circular progress meter for a 0–1 match score. The arc fills
// proportionally (0.7 → 70% of the circle) and its color shifts with quality:
// strong = green, decent = amber, weak = muted terracotta. Used on recipe
// cards as the at-a-glance "how good is this pick" mark.

interface Props {
  value: number        // 0..1
  size?: number        // px diameter
  stroke?: number      // px arc thickness
  label?: string       // small caption under the number (optional)
}

// Quality → color. Thresholds match the app's green/amber/red urgency language.
function scoreColor(pct: number): string {
  if (pct >= 75) return 'var(--green-600)'
  if (pct >= 55) return 'var(--green-500)'
  if (pct >= 35) return 'var(--amber-400)'
  return 'var(--red-400)'
}

export function ScoreRing({ value, size = 54, stroke = 5, label }: Props) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const dash = (pct / 100) * c
  const color = scoreColor(pct)

  return (
    <div style={{
      position: 'relative', width: size, height: size, flexShrink: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }} aria-hidden>
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke="var(--gray-200)" strokeWidth={stroke}
        />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke={color} strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c}`}
          style={{ transition: 'stroke-dasharray .5s cubic-bezier(.22,.8,.36,1), stroke .3s' }}
        />
      </svg>
      <div style={{
        position: 'absolute', textAlign: 'center', lineHeight: 1,
      }}>
        <div style={{ fontSize: size * 0.3, fontWeight: 700, color: 'var(--gray-800)' }}>
          {pct}
          <span style={{ fontSize: size * 0.16, fontWeight: 600, color: 'var(--gray-500)' }}>%</span>
        </div>
        {label && (
          <div style={{ fontSize: size * 0.15, color: 'var(--gray-400)', marginTop: 2, fontWeight: 600, letterSpacing: '.03em', textTransform: 'uppercase' }}>
            {label}
          </div>
        )}
      </div>
    </div>
  )
}
