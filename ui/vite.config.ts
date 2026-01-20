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
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return;
          }
          if (id.includes('/node_modules/recharts/')) {
            return 'recharts';
          }
          if (id.includes('/node_modules/lucide-react/')) {
            return 'lucide';
          }
          if (id.includes('/node_modules/@tanstack/react-query/')) {
            return 'tanstack-query';
          }
          if (id.includes('/node_modules/react-dom/')) {
            return 'react-vendor';
          }
          if (id.includes('/node_modules/react/')) {
            return 'react-vendor';
          }
          return 'vendor';
        },
      },
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
      '/prompts': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/paper-trading': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/regimes': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
    },
  },
});
