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
      // Proxy all API endpoints to the backend
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
