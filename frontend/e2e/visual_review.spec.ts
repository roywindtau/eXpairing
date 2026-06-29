/**
 * Visual review — mirrors demo.spec.ts state by state.
 * Run: npx playwright test e2e/visual_review.spec.ts --config=playwright.config.ts
 * Output: /tmp/ui-review/
 */

import { test, type Page } from '@playwright/test'
import * as path from 'path'
import * as fs from 'fs'

const OUT = '/tmp/ui-review'
fs.mkdirSync(OUT, { recursive: true })

let n = 0
const shot = async (page: Page, label: string) => {
  n++
  await page.screenshot({ path: path.join(OUT, `${String(n).padStart(2,'0')}-${label}.png`) })
}
const pause = (ms: number) => new Promise(r => setTimeout(r, ms))
const daysFromNow = (d: number) => {
  const dt = new Date(); dt.setDate(dt.getDate() + d)
  return dt.toISOString().split('T')[0]
}

test('UI visual review — demo states', async ({ page }) => {
  test.setTimeout(180_000)

  await page.goto('http://localhost:5173')
  await page.evaluate(() => localStorage.removeItem('userId'))
  await page.reload()

  // ══════════════════════════════════════════════════════════════════════
  // 1. ONBOARDING
  // ══════════════════════════════════════════════════════════════════════
  await page.getByRole('heading', { name: 'Fridge2Fork' }).waitFor({ timeout: 8000 })
  await shot(page, '01-onboarding-initial')

  await page.getByPlaceholder('e.g. Rubi').fill('Demo User')
  await shot(page, '02-onboarding-name-filled')

  await page.locator('input[type="range"]').evaluate((el: HTMLInputElement) => {
    el.value = '0.65'
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await pause(300)
  await shot(page, '03-onboarding-slider-moved')

  await page.getByRole('button', { name: 'vegetarian' }).click()
  await pause(200)
  await shot(page, '04-onboarding-vegetarian-toggled')

  await page.getByRole('button', { name: 'gluten-free' }).click()
  await pause(300)
  await shot(page, '05-onboarding-two-tags-selected')

  await page.getByRole('button', { name: /get started/i }).click()
  await page.waitForURL('**/pantry', { timeout: 10000 })
  await pause(500)

  // ══════════════════════════════════════════════════════════════════════
  // 2. PANTRY
  // ══════════════════════════════════════════════════════════════════════
  await shot(page, '06-pantry-initial-empty')

  await page.getByRole('button', { name: '+ Add item' }).click()
  await pause(200)
  await shot(page, '07-pantry-add-form-open')

  // Autocomplete demo: type "but"
  const ingredientInput = page.getByPlaceholder('e.g. milk')
  await ingredientInput.click()
  await ingredientInput.pressSequentially('but', { delay: 90 })
  await shot(page, '08-pantry-typing-but')

  await page.getByTestId('ingredient-suggestions').waitFor({ timeout: 4000 })
  await pause(400)
  await shot(page, '09-pantry-autocomplete-dropdown')

  await ingredientInput.press('ArrowDown')
  await pause(300)
  await shot(page, '10-pantry-autocomplete-item-highlighted')

  await ingredientInput.press('Enter')
  await pause(200)
  await shot(page, '11-pantry-field-filled-butter')

  await page.locator('input[type=date]').fill(daysFromNow(7))
  await pause(200)
  await shot(page, '12-pantry-form-complete')

  await page.getByRole('button', { name: 'Add' }).click()
  await pause(400)
  await shot(page, '13-pantry-one-item-added')

  for (const [name, days] of [['milk', 2], ['eggs', 4], ['lemon', 6], ['flour', 30]] as [string,number][]) {
    await page.getByRole('button', { name: '+ Add item' }).click()
    await pause(150)
    await page.getByPlaceholder('e.g. milk').fill(name)
    await page.getByPlaceholder('e.g. milk').press('Escape')
    await page.locator('input[type=date]').fill(daysFromNow(days))
    await page.getByRole('button', { name: 'Add' }).click()
    await pause(350)
  }
  await shot(page, '14-pantry-five-items-expiry-badges')

  // Vision scan
  const chooserPromise = page.waitForEvent('filechooser')
  await page.getByRole('button', { name: 'Demo scan' }).click()
  const fc = await chooserPromise
  await fc.setFiles({ name: 'fridge.jpg', mimeType: 'image/jpeg', buffer: Buffer.alloc(256) })
  await pause(700)
  await shot(page, '15-pantry-scan-loading')

  await page.getByText(/Found \d+ items/i).waitFor({ timeout: 10000 })
  await pause(400)
  await shot(page, '16-pantry-vision-review-card')

  const missingDates = page.locator('input[type="date"][value=""]')
  for (let i = 0; i < await missingDates.count(); i++) {
    await missingDates.nth(i).fill(daysFromNow(14)); await pause(100)
  }
  await pause(300)
  await shot(page, '17-pantry-vision-dates-filled')

  await page.getByRole('button', { name: /Add \d+ items to pantry/i }).click()
  await page.getByText(/items added to your pantry/i).waitFor({ timeout: 8000 })
  await pause(500)
  await shot(page, '18-pantry-items-added-toast')

  await page.getByText('eggs').first().waitFor({ timeout: 6000 })
  await pause(400)
  await shot(page, '19-pantry-full-list-after-scan')

  await page.mouse.wheel(0, 250); await pause(400)
  await shot(page, '20-pantry-scrolled-down')
  await page.mouse.wheel(0, -250); await pause(300)

  // ══════════════════════════════════════════════════════════════════════
  // 3. RECIPE FEED — cold start
  // ══════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Recipes' }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await page.locator('.card').first().waitFor({ timeout: 25000 })
  await pause(600)
  await shot(page, '21-feed-cold-start-banner')

  const firstCard = page.locator('.card').first()
  await firstCard.getByText('▼ Why this recipe?').click()
  await pause(700)
  await shot(page, '22-feed-score-breakdown-expanded')

  await firstCard.getByText('▲ Hide breakdown').click()
  await pause(300)
  await shot(page, '23-feed-score-breakdown-collapsed')

  const sortSelect = page.getByRole('combobox', { name: /sort recipes by/i })
  await sortSelect.selectOption('expiry_urgency')
  await pause(700)
  await shot(page, '24-feed-sort-expiry-urgency')

  await sortSelect.selectOption('cf_score')
  await pause(700)
  await shot(page, '25-feed-sort-cf-score')

  await sortSelect.selectOption('match_ratio')
  await pause(600)
  await shot(page, '26-feed-sort-pantry-match')

  await sortSelect.selectOption('final_score')
  await pause(500)
  await shot(page, '27-feed-sort-total-default')

  // ══════════════════════════════════════════════════════════════════════
  // 4. COOK & RATE × 5
  // ══════════════════════════════════════════════════════════════════════
  const starsToGive = [4, 5, 3, 5, 4]
  for (let i = 0; i < starsToGive.length; i++) {
    await page.locator('.card').first().waitFor({ timeout: 20000 })
    const card = page.locator('.card').first()

    await shot(page, `28-cook-${i+1}-card-before-click`)
    await card.getByRole('button', { name: '✓ Cook this' }).click()
    await pause(400)
    await shot(page, `29-cook-${i+1}-star-buttons-visible`)

    await card.locator('button').filter({ hasText: '★' }).nth(starsToGive[i] - 1).click()
    await pause(600)
    await shot(page, `30-cook-${i+1}-rated-${starsToGive[i]}stars`)

    await page.mouse.wheel(0, 260); await pause(250)
  }

  await shot(page, '31-feed-after-5-ratings-pre-refresh')

  await page.getByRole('button', { name: /refresh/i }).click()
  await page.locator('.card').first().waitFor({ timeout: 25000 })
  await pause(700)
  await shot(page, '32-feed-warm-cf-banner')

  // ══════════════════════════════════════════════════════════════════════
  // 5. RECIPE DETAIL
  // ══════════════════════════════════════════════════════════════════════
  await page.locator('.card a').first().click()
  await page.waitForURL(/\/recipe\/\d+/, { timeout: 10000 })
  await pause(500)
  await shot(page, '33-recipe-detail-top')

  await page.mouse.wheel(0, 350); await pause(400)
  await shot(page, '34-recipe-detail-ingredients')

  await page.mouse.wheel(0, 400); await pause(400)
  await shot(page, '35-recipe-detail-steps')

  await page.getByRole('button', { name: /back/i }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await pause(300)

  // ══════════════════════════════════════════════════════════════════════
  // 6. RECIPES SEARCH
  // ══════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Recipes' }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await page.locator('.card').first().waitFor({ timeout: 15000 })
  await pause(500)
  await shot(page, '36-recipes-initial')

  await page.getByPlaceholder(/search/i).fill('chicken')
  await shot(page, '37-recipes-typing-chicken')

  await pause(900)
  await shot(page, '38-recipes-search-results')

  await page.locator('.card a').first().click()
  await page.waitForURL(/\/recipe\/\d+/, { timeout: 8000 })
  await pause(400)
  await shot(page, '41-recipes-recipe-detail')

  await page.getByRole('button', { name: /back/i }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await pause(300)

  // ══════════════════════════════════════════════════════════════════════
  // 7. PROFILE
  // ══════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Profile' }).click()
  await page.waitForURL('**/profile', { timeout: 6000 })
  await pause(500)
  await shot(page, '42-profile-warm-badge-progress')

  await page.locator('input[type="range"]').evaluate((el: HTMLInputElement) => {
    el.value = '0.5'
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await pause(300)
  await shot(page, '43-profile-beta-slider-adjusted')

  await page.getByRole('button', { name: 'vegan' }).click()
  await pause(300)
  await shot(page, '44-profile-vegan-toggled')

  await page.getByRole('button', { name: /save/i }).click()
  await page.getByText(/saved/i).waitFor({ state: 'visible', timeout: 5000 })
  await pause(400)
  await shot(page, '45-profile-saved')

  // ══════════════════════════════════════════════════════════════════════
  // 8. FINAL FEED — warm + buy missing
  // ══════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'Recipes' }).click()
  await page.waitForURL('**/feed', { timeout: 6000 })
  await page.locator('.card').first().waitFor({ timeout: 25000 })
  await pause(600)
  await shot(page, '46-final-feed-overview')

  for (let i = 0; i < 4; i++) {
    await page.mouse.wheel(0, 280); await pause(300)
  }
  await shot(page, '47-final-feed-scrolled-cards')
  await page.mouse.wheel(0, -1200); await pause(500)

  const finalCard = page.locator('.card').first()
  if (await finalCard.getByText('▼ Why this recipe?').isVisible({ timeout: 3000 }).catch(() => false)) {
    await finalCard.getByText('▼ Why this recipe?').click()
    await pause(800)
    await shot(page, '48-final-feed-breakdown-warm-svd')
    await finalCard.getByText('▲ Hide breakdown').click()
    await pause(300)
  }

  const buyBtn = page.getByRole('button', { name: '＋ Buy missing' }).first()
  await buyBtn.waitFor({ timeout: 8000 })
  await shot(page, '49-final-feed-buy-missing-button')

  await buyBtn.click()
  await page.getByRole('button', { name: /Added to list/i }).first().waitFor({ timeout: 5000 })
  await pause(400)
  await shot(page, '50-final-feed-buy-missing-confirmed')

  const buyBtns2 = page.getByRole('button', { name: '＋ Buy missing' })
  if (await buyBtns2.count() > 0) {
    await buyBtns2.first().click(); await pause(500)
    await shot(page, '51-final-feed-second-recipe-added')
  }

  // ══════════════════════════════════════════════════════════════════════
  // 9. SHOPPING LIST
  // ══════════════════════════════════════════════════════════════════════
  await page.getByRole('link', { name: 'List' }).click()
  await page.waitForURL('**/list', { timeout: 6000 })
  await pause(500)
  await shot(page, '52-shopping-list-initial')

  await page.locator('input[type="checkbox"]').first().waitFor({ timeout: 6000 })
  await pause(300)
  await shot(page, '53-shopping-list-items-loaded')

  await page.mouse.wheel(0, 300); await pause(400)
  await shot(page, '54-shopping-list-scrolled')
  await page.mouse.wheel(0, -300); await pause(300)

  const checkboxes = page.locator('input[type="checkbox"]')
  await checkboxes.nth(0).click(); await pause(300)
  await shot(page, '55-shopping-list-first-checked')

  await checkboxes.nth(1).click(); await pause(400)
  await shot(page, '56-shopping-list-two-checked')

  await page.getByRole('button', { name: /Clear purchased/i }).first().click()
  await pause(600)
  await shot(page, '57-shopping-list-after-clear')
})
