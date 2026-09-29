import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  resolve: { dedupe: ["three", "react", "react-dom"] },
  server: {
    port: 5180,
    strictPort: true,
    fs: { allow: [".."] },
    proxy: { "/api": "http://127.0.0.1:8010" },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: { three: ["three"], react: ["react", "react-dom"] },
      },
    },
  },
});
