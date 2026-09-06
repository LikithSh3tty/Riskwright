import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is reached at /api/* in every environment. In development Vite
// proxies it; in the container nginx does. The frontend therefore never needs
// to know the API's host, and no VITE_* variable carries a secret.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
