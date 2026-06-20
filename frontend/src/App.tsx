import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { useEffect } from 'react'
import { useUserId } from './hooks/useUserId'
import { OnboardingPage }    from './pages/OnboardingPage'
import { PantryPage }         from './pages/PantryPage'
import { RecipeFeedPage }     from './pages/RecipeFeedPage'
import { ProfilePage }        from './pages/ProfilePage'
import { BrowsePage }         from './pages/BrowsePage'
import { RecipeDetailPage }   from './pages/RecipeDetailPage'
import { ShoppingListPage }   from './pages/ShoppingListPage'
import { WineForYouPage }     from './pages/WineForYouPage'
import { getUser } from './api/client'
import './index.css'

export default function App() {
  const { userId, setUserId, clearUserId } = useUserId()

  // Validate stored user ID against the backend on startup.
  // After a DB reset the stored ID may no longer exist — redirect to onboarding.
  useEffect(() => {
    if (!userId) return
    getUser(userId).catch(() => clearUserId())
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (!userId) {
    return (
      <BrowserRouter>
        <OnboardingPage onCreated={setUserId} />
      </BrowserRouter>
    )
  }

  return (
    <BrowserRouter>
      <div className="app-shell">
        <nav className="nav">
          <div className="nav-brand">
            <span>🍳</span> Fridge2Fork
          </div>
          <div className="nav-links">
            <NavLink to="/pantry"   className={({isActive}) => `nav-link${isActive ? ' active' : ''}`}>Pantry</NavLink>
            <NavLink to="/feed"     className={({isActive}) => `nav-link${isActive ? ' active' : ''}`}>Recipes</NavLink>
            <NavLink to="/browse"   className={({isActive}) => `nav-link${isActive ? ' active' : ''}`}>Browse</NavLink>
            <NavLink to="/wine"     className={({isActive}) => `nav-link${isActive ? ' active' : ''}`}>Wine</NavLink>
            <NavLink to="/list"     className={({isActive}) => `nav-link${isActive ? ' active' : ''}`}>List</NavLink>
            <NavLink to="/profile"  className={({isActive}) => `nav-link${isActive ? ' active' : ''}`}>Profile</NavLink>
          </div>
        </nav>

        <Routes>
          <Route path="/"        element={<Navigate to="/feed" replace />} />
          <Route path="/pantry"        element={<PantryPage      userId={userId} />} />
          <Route path="/feed"          element={<RecipeFeedPage  userId={userId} />} />
          <Route path="/browse"        element={<BrowsePage      userId={userId} />} />
          <Route path="/wine"          element={<WineForYouPage  userId={userId} />} />
          <Route path="/profile"       element={<ProfilePage     userId={userId} />} />
          <Route path="/list"          element={<ShoppingListPage userId={userId} />} />
          <Route path="/recipe/:id"    element={<RecipeDetailPage userId={userId} />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
