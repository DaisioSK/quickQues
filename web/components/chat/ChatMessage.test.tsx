/**
 * Tests for ChatMessage.
 *
 * Covers role-specific rendering + the citation list pass-through.
 * Confidence-banner branches are asserted because they're user-visible
 * trust signals — silently dropping the "low confidence" hint would be
 * a regression with real consequences.
 */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChatMessage } from "./ChatMessage";

describe("ChatMessage", () => {
  it("renders assistant content + citations", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          content: "桥梁防水由总承包方负责。",
          citations: [
            { file: "Contract DEMO.pdf", page: 12 },
            { file: "Contract DEMO.pdf", page: 14 },
          ],
          confidence: "high",
        }}
      />,
    );
    expect(screen.getByText(/桥梁防水由总承包方负责/)).toBeInTheDocument();
    // Two chips, both showing the right page number.
    expect(screen.getByText("p.12")).toBeInTheDocument();
    expect(screen.getByText("p.14")).toBeInTheDocument();
  });

  it("shows the retrieval-only banner for confidence='none'", () => {
    // Use a content string that doesn't share substring with the banner
    // text — otherwise getByText would match both nodes.
    render(
      <ChatMessage
        message={{
          role: "assistant",
          content: "示例答案",
          citations: [],
          confidence: "none",
        }}
      />,
    );
    expect(screen.getByText(/Retrieval-only mode/i)).toBeInTheDocument();
  });

  it("shows the low-confidence banner for confidence='low'", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          content: "uncertain answer",
          confidence: "low",
        }}
      />,
    );
    expect(screen.getByText(/Low confidence/i)).toBeInTheDocument();
  });

  it("renders user messages without any banner", () => {
    render(
      <ChatMessage
        message={{ role: "user", content: "我的问题" }}
      />,
    );
    expect(screen.getByText("我的问题")).toBeInTheDocument();
    expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument();
  });
});
