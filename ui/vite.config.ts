import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      // Proxy all API endpoints to the backend (ops-api on port 8081)
      '/backtests': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/live': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/market': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/agents': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/wallets': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/workflows': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/llm': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/events': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/fills': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/positions': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/status': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
    },
  },
});
