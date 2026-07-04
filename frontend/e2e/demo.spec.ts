/**
 * eXpairing — Feature Demo
 *
 * End-to-end walkthrough: Onboarding → Pantry → Demo Scan → Recipe Feed
 * → Score breakdown → Cook & Rate (x5) → Recipe Detail → Search → Profile → Shopping List
 *
 * Run:
 *   cd frontend
 *   npx playwright test e2e/demo.spec.ts --config=playwright-demo.config.ts
 *
 * Video: frontend/demo-video/
 */

import { test, type Page } from '@playwright/test'

const pause = (ms: number) => new Promise(r => setTimeout(r, ms))

function daysFromNow(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() + n)
  return d.toISOString().split('T')[0]
}

async function clearUser(page: Page) {
  await page.goto('http://localhost:5173')
  await page.evaluate(() => localStorage.removeItem('userId'))
  await page.reload()
}

async function addPantryItem(
  page: Page, ingredient: string, days: number,
  { showAutocomplete = false } = {},
) {
  await page.getByRole('button', { name: '+ Add item' }).click()
  await pause(250)
  const input = page.getByPlaceholder('e.g. milk')

  if (showAutocomplete) {
    // Type first 3 chars slowly so the dropdown appears
    await input.click()
    await input.pressSequentially(ingredient.slice(0, 3), { delay: 90 })
    // Wait for suggestions to appear, pause so viewer can see them
    await page.getByTestId('ingredient-suggestions').waitFor({ timeout: 4000 })
    await pause(800)
    // Arrow down to highlight the first suggestion, then Enter to pick it
    await input.press('ArrowDown')
    await pause(350)
    await input.press('Enter')
    await pause(300)
  } else {
    await input.fill(ingredient)
    await input.press('Escape')  // dismiss any suggestions before moving on
  }

  await page.locator('input[type=date]').fill(daysFromNow(days))
  await pause(250)
  await page.getByRole('button', { name: 'Add' }).click()
  await pause(450)
}

// ── demo ─────────────────────────────────────────────────────────────────────

test('eXpairing full feature demo', async ({ page }) => {
  test.setTimeout(180_000)

  // ══════════════════════════════════════════════════════════════════════════
  // 1. ONBOARDING
  // ══════════════════════════════════════════════════════════════════════════
  await clearUser(page)

  // App renders OnboardingPage inline at root when no userId in localStorage
  await page.getByRole('heading', { name: 'eXpairing' }).waitFor({ timeout: 8000 })
  await pause(900)

  // Fill name
  await page.getByPlaceholder('e.g. Rubi').fill('Demo User')
  await pause(600)

  // Slide waste-aversion to ~65%
  await page.locator('input[type="range"]').evaluate((el: HTMLInputElement) => {
    el.value = '0.65'
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await pause(700)

  // Toggle diet tags
  await page.getByRole('button', { name: 'vegetarian' }).click()
  await pause(350)
  await page.getByRole('button', { name: 'gluten-free' }).click()
  await pause(700)

  await page.getByRole('button', { name: /get started/i }).click()
  await page.waitForURL('**/pantry', { timeout: 10000 })
  await pause(800)

  // ══════════════════════════════════════════════════════════════════════════
  // 2. PANTRY — manually add items with urgency spread
  // ══════════════════════════════════════════════════════════════════════════
  await page.getByRole('heading', { name: 'My Pantry' }).waitFor()
  await pause(600)

  // Third item: show the autocomplete dropdown so the viewer sees it ("but" → "butter" is first suggestion)
  await addPantryItem(page, 'eggs',   4)
  await addPantryItem(page, 'milk',   2)
  await addPantryItem(page, 'butter', 7, { showAutocomplete: true })
  await addPantryItem(page, 'flour',  30)
  await addPantryItem(page, 'lemon',  6)

  // Pantry list + expiry badges visible
  await page.getByText('eggs').waitFor()
  await page.getByText('butter').waitFor()
  await pause(900)

  // ══════════════════════════════════════════════════════════════════════════
  // 3. PANTRY — demo vision scan
  // ══════════════════════════════════════════════════════════════════════════
  // "Demo scan" opens a file chooser — intercept it and supply a dummy image.
  // In demo mode the component ignores file contents and calls /vision/mock directly.
  const fileChooserPromise = page.waitForEvent('filechooser')
  await page.getByRole('button', { name: 'Demo scan' }).click()
  const fileChooser = await fileChooserPromise
  await fileChooser.setFiles({
    name: 'fridge.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.alloc(256),   // demo mode ignores the bytes
  })
  await pause(1000)

  // Wait for the review card
  await page.getByText(/Found \d+ items/i).waitFor({ timeout: 10000 })
  await pause(800)

  // Fill any missing expiry dates
  const missingInputs = page.locator('input[type="date"][value=""]')
  const missing = await missingInputs.count()
  for (let i = 0; i < missing; i++) {
    await missingInputs.nth(i).fill(daysFromNow(14))
    await pause(180)
  }
  await pause(500)

  await page.getByRole('button', { name: /Add \d+ items to pantry/i }).click()
  await page.getByText(/items added to your pantry/i).waitFor({ timeout: 8000 })
  await pause(1500)

  // Pantry updates in-place (no spinner) — all items visible now
  await page.getByText('eggs').first().waitFor({ timeout: 6000 })
  // Scroll down to show all items, then back up
  await page.mouse.wheel(0, 250)
  await pause(700)
  await page.mouse.wheel(0, -250)
  await pause(800)

  // ══════════════════════════════════════════════════════════════════════════
  // 4. RECIPE FEED — cold-start ranked recommendations
  // ══════════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Recipes' }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await pause(600)

  await page.locator('.card').first().waitFor({ timeout: 25000 })
  await pause(800)

  // Cold-start CF banner
  await page.getByText('Personalized for you (new user)').waitFor({ timeout: 8000 })
  await pause(800)

  // Score breakdown on first card
  const firstCard = page.locator('.card').first()
  await firstCard.getByText('▼ Why this recipe?').click()
  await pause(1200)
  await firstCard.getByText('▲ Hide breakdown').click()
  await pause(500)

  // ── Sort controls ────────────────────────────────────────────────────────
  const sortSelect = page.getByRole('combobox', { name: /sort recipes by/i })

  // Sort by Expiry urgency — cards reorder to show most urgent items first
  await sortSelect.selectOption('expiry_urgency')
  await pause(1000)

  // Sort by CF score — shows pure collaborative filtering ranking
  await sortSelect.selectOption('cf_score')
  await pause(1000)

  // Sort by Pantry match
  await sortSelect.selectOption('match_ratio')
  await pause(800)

  // Return to default Total score
  await sortSelect.selectOption('final_score')
  await pause(700)

  // ══════════════════════════════════════════════════════════════════════════
  // 5. COOK & RATE — 5 ratings to flip cold → warm CF
  // ══════════════════════════════════════════════════════════════════════════
  const starsToGive = [4, 5, 3, 5, 4]

  for (let i = 0; i < starsToGive.length; i++) {
    await page.locator('.card').first().waitFor({ timeout: 20000 })
    const card = page.locator('.card').first()

    await card.getByRole('button', { name: '✓ Cook this' }).click()
    await pause(600)

    // 5 ★ star buttons appear inside the card
    const starButtons = card.locator('button').filter({ hasText: '★' })
    await starButtons.nth(starsToGive[i] - 1).click()
    await pause(800)

    // Scroll to reveal next cards
    await page.mouse.wheel(0, 260)
    await pause(350)
  }

  // Click Refresh to reload feed — CF flips to warm after 5 ratings
  await page.getByRole('button', { name: /refresh/i }).click()
  await page.locator('.card').first().waitFor({ timeout: 25000 })
  await pause(1000)

  // Warm CF banner
  await page.getByText('Personalized from your history').waitFor({ timeout: 10000 })
  await pause(900)

  // ══════════════════════════════════════════════════════════════════════════
  // 6. RECIPE DETAIL
  // ══════════════════════════════════════════════════════════════════════════
  await page.locator('.card a').first().click()
  await page.waitForURL(/\/recipe\/\d+/, { timeout: 10000 })
  await pause(700)

  await page.locator('h1, h2').first().waitFor()
  await pause(500)
  await page.mouse.wheel(0, 350)   // ingredients
  await pause(700)
  await page.mouse.wheel(0, 400)   // steps
  await pause(900)

  await page.getByRole('button', { name: /back/i }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await pause(600)

  // ══════════════════════════════════════════════════════════════════════════
  // 7. SEARCH — narrow the Recipes feed + open a recipe detail
  // ══════════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Recipes' }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await pause(700)

  await page.locator('.card').first().waitFor({ timeout: 15000 })
  await pause(700)

  // Text search narrows the loaded pool by name or ingredient
  await page.getByPlaceholder(/search/i).fill('chicken')
  await pause(1200)

  // Open a recipe detail from the results
  await page.locator('.card a').first().click()
  await page.waitForURL(/\/recipe\/\d+/, { timeout: 8000 })
  await pause(700)
  await page.mouse.wheel(0, 350)
  await pause(600)
  await page.getByRole('button', { name: /back/i }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await pause(600)

  // ══════════════════════════════════════════════════════════════════════════
  // 8. PROFILE — beta slider, diet tags, CF progress
  // ══════════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Profile' }).click()
  await page.waitForURL('**/profile', { timeout: 6000 })
  await pause(700)

  // CF progress bar (5/5 → warm)
  await page.locator('[class*="progress"], [class*="cf"], .badge').first().waitFor({ timeout: 5000 })
  await pause(800)

  // Adjust beta slider
  await page.locator('input[type="range"]').evaluate((el: HTMLInputElement) => {
    el.value = '0.5'
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await pause(600)

  // Toggle vegan tag
  await page.getByRole('button', { name: 'vegan' }).click()
  await pause(400)

  await page.getByRole('button', { name: /save/i }).click()
  await page.getByText(/saved/i).waitFor({ state: 'visible', timeout: 5000 })
  await pause(1000)

  // ══════════════════════════════════════════════════════════════════════════
  // 9. FINAL FEED — warm recommendations + score breakdown
  // ══════════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Recipes' }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await page.locator('.card').first().waitFor({ timeout: 25000 })
  await pause(800)

  for (let i = 0; i < 4; i++) {
    await page.mouse.wheel(0, 280)
    await pause(400)
  }

  // Open score breakdown on first card
  const finalCard = page.locator('.card').first()
  if (await finalCard.getByText('▼ Why this recipe?').isVisible({ timeout: 3000 }).catch(() => false)) {
    await finalCard.getByText('▼ Why this recipe?').click()
    await pause(1500)
  }

  await pause(1000)

  // ══════════════════════════════════════════════════════════════════════════
  // 10. SHOPPING LIST — add missing ingredients, view list, check off, clear
  // ══════════════════════════════════════════════════════════════════════════

  // Scroll back to top so recipe cards are in view
  await page.mouse.wheel(0, -1200)
  await pause(700)

  // Find the first recipe card that has a "Buy missing" button
  const buyBtn = page.getByRole('button', { name: '＋ Buy missing' }).first()
  await buyBtn.waitFor({ timeout: 8000 })
  await pause(600)

  // Add missing ingredients to the shopping list
  await buyBtn.click()
  await page.getByRole('button', { name: /Added to list/i }).first().waitFor({ timeout: 5000 })
  await pause(900)

  // Add a second recipe's ingredients
  const buyBtns = page.getByRole('button', { name: '＋ Buy missing' })
  if (await buyBtns.count() > 0) {
    await buyBtns.first().click()
    await pause(800)
  }

  // Navigate to the Shopping List tab
  await page.getByRole('link', { name: 'List' }).click()
  await page.waitForURL('**/list', { timeout: 6000 })
  await pause(700)

  // Items visible with source recipe attribution
  await page.locator('input[type="checkbox"]').first().waitFor({ timeout: 6000 })
  await pause(600)

  // Scroll through the list
  await page.mouse.wheel(0, 300)
  await pause(700)
  await page.mouse.wheel(0, -300)
  await pause(500)

  // Check off the first two items (simulating "at the shop")
  const checkboxes = page.locator('input[type="checkbox"]')
  await checkboxes.nth(0).click()
  await pause(400)
  await checkboxes.nth(1).click()
  await pause(600)

  // Clear purchased items
  await page.getByRole('button', { name: /Clear purchased/i }).first().click()
  await pause(1000)

  await pause(800)
})
