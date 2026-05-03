import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/brainstorm': 'http://localhost:8000',
      '/ask':        'http://localhost:8000',
      '/feedback':   'http://localhost:8000',
      '/session':    'http://localhost:8000',
      '/healthz':    'http://localhost:8000',
    },
  },
})
