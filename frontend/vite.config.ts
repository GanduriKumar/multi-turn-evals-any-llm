import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/datasets': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/runs': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/goldens': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/version': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/settings': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/compare': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/validate': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/metrics-config': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/coverage': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/reports': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    }
  }
})
