import { defineConfig } from '@playwright/test'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

export default defineConfig({
  testDir: './e2e',
  testMatch: 'demo.spec.ts',
  timeout: 180_000,
  retries: 0,
  use: {
    baseURL: 'http://localhost:5173',
    headless: true,
    viewport: { width: 1280, height: 800 },
    slowMo: 120,
    video: {
      mode: 'on',
      size: { width: 1280, height: 800 },
    },
    screenshot: 'on',
  },
  outputDir: join(__dirname, 'demo-video'),
  reporter: [['list']],
})
