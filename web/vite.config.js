import { defineConfig } from "vite";

export default defineConfig({
  root: __dirname,
  base: "/",
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 40000, // transformers.js 体积较大，仅提示阈值
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8787", changeOrigin: true },
    },
  },
});
