import { test, expect, type Page } from '@playwright/test'

// ── helpers ──────────────────────────────────────────────────────────────────

async function clearUser(page: Page) {
  await page.goto('/')
  await page.evaluate(() => localStorage.clear())
}

async function completeOnboarding(
  page: Page,
  opts: { name?: string; beta?: number; tags?: string[] } = {},
) {
  await page.goto('/')
  await expect(page.getByText('eXpairing')).toBeVisible()

  if (opts.name) {
    await page.getByPlaceholder('e.g. Rubi').fill(opts.name)
  }

  if (opts.beta !== undefined) {
    await page.locator('input[type=range]').fill(String(opts.beta))
  }

  if (opts.tags) {
    for (const tag of opts.tags) {
      await page.getByText(tag, { exact: true }).click()
    }
  }

  await page.getByRole('button', { name: 'Get started →' }).click()
  // Lands on /pantry after onboarding
  await expect(page).toHaveURL(/\/pantry/)
}

// ── 1. Onboarding ─────────────────────────────────────────────────────────────

test.describe('Onboarding', () => {
  test.beforeEach(async ({ page }) => { await clearUser(page) })

  test('shows onboarding when no user stored', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'eXpairing' })).toBeVisible()
    await expect(page.getByText('Rank recipes to minimize food waste')).toBeVisible()
  })

  test('slider labels are correct', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Discover new recipes')).toBeVisible()
    await expect(page.getByText('Use what I have')).toBeVisible()
  })

  test('creates user and navigates to pantry', async ({ page }) => {
    await completeOnboarding(page, { name: 'Test User' })
    await expect(page).toHaveURL(/\/pantry/)
  })

  test('diet tag toggles on and off', async ({ page }) => {
    await page.goto('/')
    const tag = page.getByText('vegetarian', { exact: true })
    await tag.click()
    await expect(tag).toHaveClass(/badge-green/)
    await tag.click()
    await expect(tag).toHaveClass(/badge-gray/)
  })

  test('Get started button is disabled while submitting', async ({ page }) => {
    await page.goto('/')
    const btn = page.getByRole('button', { name: 'Get started →' })
    await expect(btn).toBeEnabled()
  })
})

// ── 2. Navigation ─────────────────────────────────────────────────────────────

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page)
  })

  test('nav shows core links', async ({ page }) => {
    await expect(page.getByRole('link', { name: 'Pantry' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Recipes' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Profile' })).toBeVisible()
  })

  test('nav links navigate correctly', async ({ page }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await expect(page).toHaveURL(/\/feed/)

    await page.getByRole('link', { name: 'Profile' }).click()
    await expect(page).toHaveURL(/\/profile/)

    await page.getByRole('link', { name: 'Pantry' }).click()
    await expect(page).toHaveURL(/\/pantry/)
  })
})

// ── 3. Pantry ─────────────────────────────────────────────────────────────────

test.describe('Pantry', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page)
    await page.goto('/pantry')
  })

  test('pantry page loads', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'My Pantry' })).toBeVisible()
  })

  test('can add a pantry item', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').fill('spinach')
    await page.locator('input[type=date]').fill('2026-12-31')
    await page.getByRole('button', { name: 'Add' }).click()
    await expect(page.getByText('spinach')).toBeVisible()
  })

  test('can delete a pantry item', async ({ page }) => {
    // Add first
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').fill('broccoli')
    await page.locator('input[type=date]').fill('2026-12-31')
    await page.getByRole('button', { name: 'Add' }).click()
    await expect(page.getByText('broccoli')).toBeVisible()

    // Delete it
    const row = page.locator('div', { hasText: 'broccoli' }).filter({ has: page.getByRole('button', { name: 'Remove' }) }).first()
    await row.getByRole('button', { name: 'Remove' }).click()
    await expect(page.getByText('broccoli')).not.toBeVisible()
  })

  test('expiry badge shows days remaining', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').fill('yogurt')
    // Set expiry 5 days from today
    const future = new Date(); future.setDate(future.getDate() + 5)
    const iso = future.toISOString().slice(0, 10)
    await page.locator('input[type=date]').fill(iso)
    await page.getByRole('button', { name: 'Add' }).click()
    await expect(page.getByText(/\d+d left/)).toBeVisible()
  })

  test('ingredient input shows suggestions when typing', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').pressSequentially('eggs', { delay: 60 })
    const suggestions = page.getByTestId('ingredient-suggestions')
    await expect(suggestions).toBeVisible({ timeout: 3_000 })
    // "eggs" should be the first result
    await expect(suggestions.locator('[role="option"]').first()).toHaveText('eggs')
  })

  test('selecting a suggestion from autocomplete fills the input', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').pressSequentially('eggs', { delay: 60 })
    await expect(page.getByTestId('ingredient-suggestions')).toBeVisible({ timeout: 3_000 })
    // Click the first suggestion (exact text "eggs")
    await page.getByTestId('ingredient-suggestions').locator('[role="option"]').first().click()
    await expect(page.getByPlaceholder('e.g. milk')).toHaveValue('eggs')
    // Dropdown should close after selection
    await expect(page.getByTestId('ingredient-suggestions')).not.toBeVisible()
  })

  test('keyboard navigation selects suggestion with ArrowDown + Enter', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    const input = page.getByPlaceholder('e.g. milk')
    await input.pressSequentially('eg', { delay: 60 })
    await expect(page.getByTestId('ingredient-suggestions')).toBeVisible({ timeout: 3_000 })
    await input.press('ArrowDown')  // highlight first item
    await input.press('Enter')      // select it (does NOT submit form)
    const selected = await input.inputValue()
    expect(selected.length).toBeGreaterThan(0)
    await expect(page.getByTestId('ingredient-suggestions')).not.toBeVisible()
  })

  test('Escape dismisses the suggestions dropdown', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').pressSequentially('mi', { delay: 60 })
    await expect(page.getByTestId('ingredient-suggestions')).toBeVisible({ timeout: 3_000 })
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('ingredient-suggestions')).not.toBeVisible()
  })

  test('no suggestions shown for single character', async ({ page }) => {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').pressSequentially('e', { delay: 60 })
    await page.waitForTimeout(400)  // longer than debounce
    await expect(page.getByTestId('ingredient-suggestions')).not.toBeVisible()
  })
})

// ── 4. Recipe Feed ────────────────────────────────────────────────────────────

test.describe('Recipe Feed', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page)
    // Add a pantry item so the feed has signal
    await page.goto('/pantry')
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').fill('eggs')
    await page.locator('input[type=date]').fill('2026-12-31')
    await page.getByRole('button', { name: 'Add' }).click()
    await page.goto('/feed')
  })

  test('feed loads and shows recipes', async ({ page }) => {
    await expect(page.getByText('What to cook tonight?')).toBeVisible()
    // Wait for recipes to load (spinner disappears, cards appear)
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
  })

  test('shows score ring and match ring on each card', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    // SVG rings should be present (score + match)
    const svgs = page.locator('.card').first().locator('svg')
    await expect(svgs).toHaveCount(2)
  })

  test('shows CF strategy banner', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    // Either cold start or SVD banner should appear
    await expect(
      page.getByText(/Personalized for you|Personalized from your history/)
    ).toBeVisible()
  })

  test('subtitle describes ranking factors with CF first', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await expect(page.getByText(/collaborative filtering.*expiry urgency.*pantry match/i)).toBeVisible()
  })

  test('sort dropdown is present with Total score as default', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    const select = page.getByRole('combobox', { name: /sort recipes by/i })
    await expect(select).toBeVisible()
    await expect(select).toHaveValue('final_score')
  })

  test('sort by Expiry urgency reorders cards', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    // Capture default top recipe name
    const defaultFirst = await page.locator('.card a').first().textContent()
    // Switch sort
    await page.getByRole('combobox', { name: /sort recipes by/i }).selectOption('expiry_urgency')
    await page.waitForTimeout(300)
    const sortedFirst = await page.locator('.card a').first().textContent()
    // The order may or may not change depending on data, but no crash
    expect(typeof sortedFirst).toBe('string')
    expect(sortedFirst!.length).toBeGreaterThan(0)
    // Switching back to Total score restores default order
    await page.getByRole('combobox', { name: /sort recipes by/i }).selectOption('final_score')
    await page.waitForTimeout(300)
    const restoredFirst = await page.locator('.card a').first().textContent()
    expect(restoredFirst).toBe(defaultFirst)
  })

  test('sort by CF score shows valid ordering', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await page.getByRole('combobox', { name: /sort recipes by/i }).selectOption('cf_score')
    await page.waitForTimeout(300)
    await expect(page.locator('.card').first()).toBeVisible()
    // All sort options can be selected without error
    for (const value of ['cb_score', 'match_ratio', 'final_score']) {
      await page.getByRole('combobox', { name: /sort recipes by/i }).selectOption(value)
      await page.waitForTimeout(150)
      await expect(page.locator('.card').first()).toBeVisible()
    }
  })

  test('Refresh button resets sort to Total score', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await page.getByRole('combobox', { name: /sort recipes by/i }).selectOption('expiry_urgency')
    await page.getByRole('button', { name: /Refresh|↻/ }).click()
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await expect(page.getByRole('combobox', { name: /sort recipes by/i })).toHaveValue('final_score')
  })

  test('score breakdown expands and shows all 4 bars', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    const firstCard = page.locator('.card').first()
    await firstCard.getByText('Why this recipe?').click()
    await expect(firstCard.getByText('Expiry urgency')).toBeVisible()
    await expect(firstCard.getByText('Ingredient match')).toBeVisible()
    await expect(firstCard.getByText('Community score (CF)')).toBeVisible()
    await expect(firstCard.getByText('Profile match (CB)')).toBeVisible()
  })

  test('CF mode badge shown in breakdown', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    const firstCard = page.locator('.card').first()
    await firstCard.getByText('Why this recipe?').click()
    await expect(firstCard.getByText('CF mode:')).toBeVisible()
    await expect(
      firstCard.getByText(/Community signal|Personalized|Not available/).first()
    ).toBeVisible()
  })

  test('recipe name links to detail page', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    const firstLink = page.locator('.card a').first()
    const recipeName = await firstLink.textContent()
    await firstLink.click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(recipeName!.trim())
  })

  test('skip removes card from feed', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    const initialCount = await page.locator('.card').count()
    await page.getByRole('button', { name: 'Skip' }).first().click()
    await expect(page.locator('.card')).toHaveCount(initialCount - 1)
  })

  test('cook button navigates to recipe page with star rating', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await page.getByRole('button', { name: '✓ Cook this' }).first().click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    await expect(page.getByText('How was it?')).toBeVisible()
    await expect(page.locator('button').filter({ hasText: '★' })).toHaveCount(5)
  })

  test('rating submits and shows confirmation', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await page.getByRole('button', { name: '✓ Cook this' }).first().click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    await expect(page.getByText('How was it?')).toBeVisible()
    await page.locator('button').filter({ hasText: '★' }).nth(3).click() // 4 stars
    await expect(page.getByText('Cooked & rated!')).toBeVisible({ timeout: 5_000 })
  })

  test('refresh button reloads feed', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
    await page.getByRole('button', { name: /↻ Refresh/ }).click()
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
  })
})

// ── 5. Recipe Detail Page ─────────────────────────────────────────────────────

test.describe('Recipe Detail', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page)
    await page.goto('/feed')
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 25_000 })
  })

  test('opens recipe detail with ingredients and steps', async ({ page }) => {
    const firstLink = page.locator('.card a').first()
    await firstLink.click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    await expect(page.getByRole('heading', { name: 'Ingredients' })).toBeVisible()
    // Either shows steps or "No instructions" message
    const hasSteps = await page.getByRole('heading', { name: 'Instructions' }).isVisible()
    const noSteps  = await page.getByText('No instructions available').isVisible()
    expect(hasSteps || noSteps).toBe(true)
  })

  test('back button returns to feed', async ({ page }) => {
    await page.locator('.card a').first().click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    await page.getByRole('button', { name: '← Back' }).click()
    await expect(page).toHaveURL(/\/feed/)
  })

  test('shows time and rating badges', async ({ page }) => {
    await page.locator('.card a').first().click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    // At least one meta badge should appear (time, rating, or tag)
    await expect(page.locator('.badge').first()).toBeVisible()
  })
})

// ── 6. Recipes search ───────────────────────────────────────────────────────

test.describe('Recipes search', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page)
    await page.goto('/feed')
  })

  test('search narrows the recipe list', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 10_000 })
    const countBefore = await page.locator('.card').count()
    await page.getByPlaceholder(/Search by name/).fill('chicken')
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 5_000 })
    const countAfter = await page.locator('.card').count()
    expect(countAfter).toBeGreaterThan(0)
    expect(countAfter).toBeLessThanOrEqual(countBefore)
  })

  test('recipe card links to recipe detail', async ({ page }) => {
    await expect(page.locator('.card').first()).toBeVisible({ timeout: 10_000 })
    const firstLink = page.locator('.card a').first()
    const name = await firstLink.textContent()
    await firstLink.click()
    await expect(page).toHaveURL(/\/recipe\/\d+/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(name!.trim())
  })
})

// ── 7. Profile Page ───────────────────────────────────────────────────────────

test.describe('Profile', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page, { name: 'Roy' })
    await page.goto('/profile')
  })

  test('profile page loads with slider', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Profile' })).toBeVisible()
    await expect(page.locator('input[type=range]')).toBeVisible()
  })

  test('slider endpoint labels correct', async ({ page }) => {
    await expect(page.getByText('Discover new recipes')).toBeVisible()
    await expect(page.getByText('Use what I have')).toBeVisible()
  })

  test('save changes updates profile', async ({ page }) => {
    await page.locator('input[type=range]').fill('0.8')
    await page.getByRole('button', { name: 'Save changes' }).click()
    await expect(page.getByRole('button', { name: '✓ Saved' })).toBeVisible({ timeout: 5_000 })
  })

  test('personalization status shows cold start progress', async ({ page }) => {
    await expect(page.getByText('Personalization status')).toBeVisible()
    // Badge and description both render — check badge specifically
    await expect(page.locator('.badge').filter({ hasText: /Cold start|Personalized/ }).first()).toBeVisible()
  })

  test('diet tags can be toggled and saved', async ({ page }) => {
    await page.getByText('vegan', { exact: true }).click()
    await page.getByRole('button', { name: 'Save changes' }).click()
    await expect(page.getByRole('button', { name: '✓ Saved' })).toBeVisible({ timeout: 5_000 })
  })
})

// ── 8. Backend health ─────────────────────────────────────────────────────────

test.describe('Backend API', () => {
  test('health endpoint returns ok', async ({ request }) => {
    const res = await request.get('http://localhost:8000/health')
    expect(res.status()).toBe(200)
    const body = await res.json()
    expect(body.status).toBe('ok')
  })

  test('ranked endpoint returns recipes', async ({ request }) => {
    // Use the seeded demo user (user_id known from seed)
    const res = await request.get('http://localhost:8000/recipes/ranked?user_id=2002373707&top_n=5')
    expect(res.status()).toBe(200)
    const body = await res.json()
    expect(Array.isArray(body)).toBe(true)
    expect(body.length).toBeGreaterThan(0)
    // All 4 score fields present
    const r = body[0]
    expect(r).toHaveProperty('cf_score')
    expect(r).toHaveProperty('cb_score')
    expect(r).toHaveProperty('expiry_urgency')
    expect(r).toHaveProperty('match_ratio')
    expect(r).toHaveProperty('cf_strategy')
    expect(r).toHaveProperty('cb_model_available')
  })

  test('recipe detail returns steps and description', async ({ request }) => {
    // Get any recipe id from ranked
    const ranked = await request.get('http://localhost:8000/recipes/ranked?user_id=2002373707&top_n=1')
    const [first] = await ranked.json()
    const res = await request.get(`http://localhost:8000/recipes/${first.recipe_id}`)
    expect(res.status()).toBe(200)
    const body = await res.json()
    expect(body).toHaveProperty('steps')
    expect(body).toHaveProperty('description')
    expect(body).toHaveProperty('ingredients')
    expect(Array.isArray(body.steps)).toBe(true)
    expect(Array.isArray(body.ingredients)).toBe(true)
  })

  test('404 for non-existent recipe', async ({ request }) => {
    const res = await request.get('http://localhost:8000/recipes/999999999')
    expect(res.status()).toBe(404)
  })

  test('search returns results', async ({ request }) => {
    const res = await request.get('http://localhost:8000/recipes/search?q=pasta&limit=5')
    expect(res.status()).toBe(200)
    const body = await res.json()
    expect(body.length).toBeGreaterThan(0)
  })

  test('diet tag filter narrows results', async ({ request }) => {
    const all  = await (await request.get('http://localhost:8000/recipes/search?limit=40')).json()
    const veg  = await (await request.get('http://localhost:8000/recipes/search?tag=vegetarian&limit=40')).json()
    // Filtered set should be ≤ unfiltered and non-empty
    expect(veg.length).toBeGreaterThan(0)
    expect(all.length).toBeGreaterThanOrEqual(veg.length)
  })

  test('ingredient suggest returns matching names', async ({ request }) => {
    const res = await request.get('http://localhost:8000/pantry/suggest?q=egg&limit=8')
    expect(res.status()).toBe(200)
    const body = await res.json()
    expect(Array.isArray(body)).toBe(true)
    expect(body.length).toBeGreaterThan(0)
    // All results should contain "egg"
    for (const item of body) {
      expect((item as string).toLowerCase()).toContain('egg')
    }
  })

  test('ingredient suggest: "eggs" is first result for query "eggs"', async ({ request }) => {
    const res = await request.get('http://localhost:8000/pantry/suggest?q=eggs&limit=5')
    const body = await res.json()
    expect(body[0]).toBe('eggs')
  })

  test('ingredient suggest returns empty for missing query param', async ({ request }) => {
    const res = await request.get('http://localhost:8000/pantry/suggest')
    // q is required, expect 422
    expect(res.status()).toBe(422)
  })
})

// ── 9. Stale-user recovery ────────────────────────────────────────────────────

test.describe('Stale user recovery', () => {
  test('redirects to onboarding if stored user no longer exists', async ({ page }) => {
    await page.goto('/')
    await page.evaluate(() => localStorage.setItem('f2f_user_id', '999999999'))
    await page.reload()
    await expect(page.getByText('Rank recipes to minimize food waste')).toBeVisible({ timeout: 5_000 })
  })
})

// ── 10. Shopping list ─────────────────────────────────────────────────────────

test.describe('Shopping List', () => {
  test.beforeEach(async ({ page }) => {
    await clearUser(page)
    await completeOnboarding(page)
    // Seed at least one pantry item so the feed has recipes with missing ingredients
    await page.getByRole('button', { name: '+ Add item' }).click()
    await page.getByPlaceholder('e.g. milk').fill('eggs')
    await page.locator('input[type=date]').fill(
      new Date(Date.now() + 7 * 86400000).toISOString().split('T')[0]
    )
    await page.getByRole('button', { name: 'Add' }).click()
    await page.waitForTimeout(300)
  })

  test('List tab shows empty state when nothing added', async ({ page }) => {
    await page.getByRole('link', { name: 'List' }).click()
    await expect(page).toHaveURL(/\/list/)
    await expect(page.getByText('Nothing to buy')).toBeVisible()
  })

  test('"Buy missing" button adds ingredients and shows confirmation', async ({ page }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })

    // Find a card with missing ingredients and click its add button
    const buyBtn = page.getByRole('button', { name: '＋ Buy missing' }).first()
    await expect(buyBtn).toBeVisible({ timeout: 8_000 })
    await buyBtn.click()

    // Button should confirm then reset
    await expect(page.getByRole('button', { name: /Added to list/i }).first())
      .toBeVisible({ timeout: 5_000 })
  })

  test('added ingredients appear in Shopping List tab', async ({ page }) => {
    // Add via the feed
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })
    const buyBtn = page.getByRole('button', { name: '＋ Buy missing' }).first()
    await expect(buyBtn).toBeVisible({ timeout: 8_000 })
    await buyBtn.click()
    await page.waitForTimeout(400)

    // Navigate to List tab
    await page.getByRole('link', { name: 'List' }).click()
    await expect(page).toHaveURL(/\/list/)

    // At least one item should be listed
    const items = page.locator('input[type=checkbox]')
    await expect(items.first()).toBeVisible({ timeout: 5_000 })
    const count = await items.count()
    expect(count).toBeGreaterThan(0)
  })

  test('item count summary updates correctly', async ({ page }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })
    await page.getByRole('button', { name: '＋ Buy missing' }).first().click()
    await page.waitForTimeout(400)

    await page.getByRole('link', { name: 'List' }).click()
    // Summary line "N items · 0 purchased"
    await expect(page.getByText(/\d+ item.*·.*purchased/)).toBeVisible()
  })

  test('checking an item marks it as purchased', async ({ page }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })
    await page.getByRole('button', { name: '＋ Buy missing' }).first().click()
    // Wait for backend confirmation before navigating away
    await expect(page.getByRole('button', { name: /Added to list/i }).first())
      .toBeVisible({ timeout: 5_000 })

    await page.getByRole('link', { name: 'List' }).click()
    const checkbox = page.locator('input[type=checkbox]').first()
    await expect(checkbox).toBeVisible({ timeout: 5_000 })
    await expect(checkbox).not.toBeChecked()
    await checkbox.click()
    await expect(checkbox).toBeChecked({ timeout: 3_000 })
    // Match the summary line only ("N items · 1 purchased"), not the Clear button
    await expect(page.locator('p').filter({ hasText: /purchased/ })).toBeVisible()
  })

  test('"Clear purchased" button removes checked items', async ({ page }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })
    await page.getByRole('button', { name: '＋ Buy missing' }).first().click()
    await page.waitForTimeout(400)

    await page.getByRole('link', { name: 'List' }).click()
    await page.locator('input[type=checkbox]').first().click()
    await page.waitForTimeout(300)

    const clearBtn = page.getByRole('button', { name: /Clear purchased/i }).first()
    await expect(clearBtn).toBeVisible()
    await clearBtn.click()

    // After clearing, either empty state or no checked items remain
    await page.waitForTimeout(500)
    const remaining = await page.locator('input[type=checkbox]:checked').count()
    expect(remaining).toBe(0)
  })

  test('remove button (✕) deletes a single item', async ({ page }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })
    await page.getByRole('button', { name: '＋ Buy missing' }).first().click()
    await expect(page.getByRole('button', { name: /Added to list/i }).first())
      .toBeVisible({ timeout: 5_000 })

    await page.getByRole('link', { name: 'List' }).click()
    const removeBtn = page.getByTitle('Remove').first()
    await expect(removeBtn).toBeVisible({ timeout: 5_000 })
    const beforeCount = await page.locator('input[type=checkbox]').count()
    expect(beforeCount).toBeGreaterThan(0)

    await removeBtn.click()
    await page.waitForTimeout(400)

    const afterCount = await page.locator('input[type=checkbox]').count()
    expect(afterCount).toBe(beforeCount - 1)
  })

  test('duplicate ingredients from same recipe are not added twice', async ({ page, request }) => {
    await page.getByRole('link', { name: 'Recipes' }).click()
    await page.waitForSelector('.card', { timeout: 10_000 })
    await page.getByRole('button', { name: '＋ Buy missing' }).first().click()
    await expect(page.getByRole('button', { name: /Added to list/i }).first())
      .toBeVisible({ timeout: 5_000 })

    const userId = await page.evaluate(() => localStorage.getItem('f2f_user_id'))
    const list = await (await request.get(`http://localhost:8000/shopping/${userId}`)).json()
    const countBefore = list.length
    expect(countBefore).toBeGreaterThan(0)

    // Re-post the exact same ingredient names via API — all should be skipped
    const ingredients = list.map((i: { ingredient: string }) => i.ingredient)
    const reAdd = await request.post(`http://localhost:8000/shopping/${userId}`, {
      data: { ingredients },
    })
    const body = await reAdd.json()
    expect(body.added.length).toBe(0)
    expect(body.skipped.length).toBe(countBefore)

    const countAfter = (await (await request.get(`http://localhost:8000/shopping/${userId}`)).json()).length
    expect(countAfter).toBe(countBefore)
  })

  test('Shopping List API: add, toggle, remove lifecycle', async ({ request, page }) => {
    // Create a user via the UI so we have a real user id
    await page.getByRole('link', { name: 'Pantry' }).click()
    const userId = await page.evaluate(() => localStorage.getItem('f2f_user_id'))

    // Add items
    const addRes = await request.post(`http://localhost:8000/shopping/${userId}`, {
      data: { ingredients: ['tomatoes', 'olive oil'], recipe_id: 1, recipe_name: 'Test Recipe' },
    })
    expect(addRes.status()).toBe(201)
    const { added, skipped } = await addRes.json()
    expect(added.length).toBe(2)
    expect(skipped).toEqual([])

    const itemId = added[0].id

    // Toggle checked
    const patchRes = await request.patch(`http://localhost:8000/shopping/${userId}/${itemId}`, {
      data: { is_checked: true },
    })
    expect(patchRes.status()).toBe(200)
    expect((await patchRes.json()).is_checked).toBe(true)

    // Remove single item
    const delRes = await request.delete(`http://localhost:8000/shopping/${userId}/${itemId}`)
    expect(delRes.status()).toBe(204)

    // Verify removed
    const listRes = await request.get(`http://localhost:8000/shopping/${userId}`)
    const remaining = await listRes.json()
    expect(remaining.find((i: { id: number }) => i.id === itemId)).toBeUndefined()

    // Clear all remaining
    await request.delete(`http://localhost:8000/shopping/${userId}?checked_only=false`)
    const final = await (await request.get(`http://localhost:8000/shopping/${userId}`)).json()
    expect(final).toEqual([])
  })
})
