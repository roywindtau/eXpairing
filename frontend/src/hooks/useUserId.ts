import { useState } from 'react'

const KEY = 'f2f_user_id'

export function useUserId() {
  const [userId, setUserIdState] = useState<number | null>(() => {
    const stored = localStorage.getItem(KEY)
    return stored ? parseInt(stored, 10) : null
  })

  const setUserId = (id: number) => {
    localStorage.setItem(KEY, String(id))
    setUserIdState(id)
  }

  const clearUserId = () => {
    localStorage.removeItem(KEY)
    setUserIdState(null)
  }

  return { userId, setUserId, clearUserId }
}
