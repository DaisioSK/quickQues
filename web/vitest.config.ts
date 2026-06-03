/**
 * Vitest config for the j-contract web/ frontend.
 *
 * Why a separate vitest config (not piggyback on next.config.ts):
 *   Next has its own jest-style test runner story (`next/jest`) but it
 *   targets jest, not vitest. We chose vitest per DECISION-5.next.3
 *   (ESM-native, faster, Vite ecosystem). The two configs don't fight.
 *
 * Test environment = jsdom because every test we write today touches
 * React components — RTL needs a DOM. If the surface grows to pure-JS
 * utility tests (api-client, format helpers) we can override per-file
 * with `@vitest-environment node` comments rather than juggle two
 * config files.
 *
 * Path alias `@/*` mirrors tsconfig.json so `import { Button } from
 * "@/components/ui/Button"` resolves the same way in source and tests.
 */

import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    css: false, // we don't need to evaluate Tailwind in tests
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
