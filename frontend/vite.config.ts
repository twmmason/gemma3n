import { defineConfig } from 'vite';

export default defineConfig({
  define: { 'process.env': {} }, // silence Node polyfills
  build: { target: 'es2022' },
  plugins: []                    // add Vite plugins here
});