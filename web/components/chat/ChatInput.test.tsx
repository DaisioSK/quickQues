/**
 * Tests for ChatInput.
 *
 * Focus on the submit semantics (Enter / Shift+Enter / Send button)
 * because those are the contract callers depend on. We don't assert on
 * disabled visual styling — that's the Button atom's contract, already
 * tested in components/ui/Button.test.tsx.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ChatInput } from "./ChatInput";

describe("ChatInput", () => {
  it("submits the trimmed question when Enter is pressed (no Shift)", () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);
    const textarea = screen.getByRole("textbox", { name: /question/i });
    fireEvent.change(textarea, { target: { value: "  桥梁防水谁负责?  " } });
    fireEvent.keyDown(textarea, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledExactlyOnceWith("桥梁防水谁负责?");
  });

  it("inserts a newline (does NOT submit) when Shift+Enter is pressed", () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);
    const textarea = screen.getByRole("textbox", { name: /question/i });
    fireEvent.change(textarea, { target: { value: "line 1" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("does not submit empty / whitespace-only text", () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} />);
    const textarea = screen.getByRole("textbox", { name: /question/i });
    fireEvent.change(textarea, { target: { value: "   " } });
    fireEvent.keyDown(textarea, { key: "Enter" });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("does not submit when disabled (e.g., request in flight)", () => {
    const onSubmit = vi.fn();
    render(<ChatInput onSubmit={onSubmit} disabled />);
    const textarea = screen.getByRole("textbox", { name: /question/i });
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter" });
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
