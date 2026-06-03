/**
 * Tests for SuggestedQuestions (empty-state example prompts).
 *
 * Contract this component must respect:
 *   - Renders one clickable chip per DEFAULT_SUGGESTIONS entry.
 *   - Clicking a chip calls onPick with that exact question text so the
 *     parent can submit it as a turn (same string the user would type).
 *   - `disabled` disables every chip (in-flight request guard).
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { DEFAULT_SUGGESTIONS, SuggestedQuestions } from "./SuggestedQuestions";

describe("SuggestedQuestions", () => {
  it("renders one chip per default suggestion", () => {
    render(<SuggestedQuestions onPick={vi.fn()} />);
    for (const q of DEFAULT_SUGGESTIONS) {
      expect(screen.getByRole("button", { name: q })).toBeInTheDocument();
    }
  });

  it("calls onPick with the exact question text when a chip is clicked", () => {
    const onPick = vi.fn();
    render(<SuggestedQuestions onPick={onPick} />);
    fireEvent.click(screen.getByRole("button", { name: DEFAULT_SUGGESTIONS[0] }));
    expect(onPick).toHaveBeenCalledExactlyOnceWith(DEFAULT_SUGGESTIONS[0]);
  });

  it("disables every chip while disabled", () => {
    render(<SuggestedQuestions onPick={vi.fn()} disabled />);
    for (const q of DEFAULT_SUGGESTIONS) {
      expect(screen.getByRole("button", { name: q })).toBeDisabled();
    }
  });

  it("renders a custom questions set when provided (domain-agnostic)", () => {
    const custom = ["这家公司去年的营收是多少？", "审计意见是什么？"];
    render(<SuggestedQuestions onPick={vi.fn()} questions={custom} />);
    for (const q of custom) {
      expect(screen.getByRole("button", { name: q })).toBeInTheDocument();
    }
    // The default DEMO set is NOT shown when overridden.
    expect(screen.queryByRole("button", { name: DEFAULT_SUGGESTIONS[0] })).toBeNull();
  });
});
