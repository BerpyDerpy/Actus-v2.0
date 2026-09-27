import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Built to static files (`npm run build` -> ui/dist) and served offline by actus/app/server.py.
// The dev server proxies /api, /ws and /stream to the FastAPI backend (`uv run python -m
// actus.app.server --replay correct`, default port 8000) so `npm run dev` works standalone.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/stream": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
