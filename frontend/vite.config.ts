import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,                  // listen on 0.0.0.0 so Docker can expose it
    watch: { usePolling: true }, // bind mounts (macOS→Linux) miss fs events otherwise
  },
})
