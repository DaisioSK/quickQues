/**
 * Integration test for the chat page.
 *
 * Mocks `@/lib/api-client` to bypass the network — the page is what
 * we're testing, not the HTTP wrapper (which has its own tests in
 * lib/api-client.test.ts).
 *
 * Covers the happy path: submit → loading banner → answer rendered
 * with citations. Failure paths (askQuestion throws → error banner)
 * are documented in the page but not asserted here — keeping this test
 * focused on the contract ssWire will visually verify next.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// Hoisted mock — must declare before importing Home, since Home imports
// askQuestion at module level.
vi.mock("@/lib/api-client", () => ({
  askQuestion: vi.fn(),
}));

import { askQuestion } from "@/lib/api-client";
import Home from "./page";

describe("chat page", () => {
  it("submits a question and renders the answer + citations", async () => {
    vi.mocked(askQuestion).mockResolvedValue({
      answer: "桥梁防水由总承包方负责。",
      citations: [{ file: "Contract DEMO.pdf", page: 12 }],
      confidence: "high",
    });

    render(<Home />);

    const textarea = screen.getByRole("textbox", { name: /question/i });
    fireEvent.change(textarea, { target: { value: "桥梁防水谁负责?" } });
    fireEvent.keyDown(textarea, { key: "Enter" });

    // Loading state renders the "Thinking…" banner immediately.
    expect(screen.getByText(/Thinking/i)).toBeInTheDocument();

    // Once the mock resolves the answer + citation should appear.
    await waitFor(() => {
      expect(screen.getByText(/桥梁防水由总承包方负责/)).toBeInTheDocument();
    });
    expect(screen.getByText("p.12")).toBeInTheDocument();

    // Loading banner clears after the response.
    expect(screen.queryByText(/Thinking/i)).not.toBeInTheDocument();

    // askQuestion was called exactly once with the trimmed text.
    expect(askQuestion).toHaveBeenCalledExactlyOnceWith("桥梁防水谁负责?");
  });
});
