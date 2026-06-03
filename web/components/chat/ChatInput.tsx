/**
 * ChatInput — textarea + send button used to ask a question.
 *
 * Submit semantics:
 *   - Enter (no Shift)  → submit
 *   - Shift+Enter       → newline
 *   - Click Send button → submit
 *
 * Why max=1000 enforced client-side too:
 *   The backend Pydantic schema (AskRequest in api/schemas.py) already
 *   caps at 1000. Mirroring it here gives a friendly UI block at the
 *   textarea boundary rather than a 422 round-trip — and stays in sync
 *   because the constant lives in one place (we just hardcode 1000;
 *   when this number changes it's one find-and-replace).
 */

"use client";

import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/Button";
import { TextArea } from "@/components/ui/Input";

const MAX_QUESTION_CHARS = 1000;

interface ChatInputProps {
  /** Called with the trimmed question string when the user submits. */
  onSubmit: (text: string) => void;
  /** When true, input + button are disabled (e.g., request in flight). */
  disabled?: boolean;
}

export function ChatInput({ onSubmit, disabled = false }: ChatInputProps) {
  const [text, setText] = useState("");

  function trySubmit() {
    const trimmed = text.trim();
    // Defense in depth: ignore submission when disabled even if a Form's
    // submit somehow fires (browser autofill quirks, screen readers, etc.)
    if (disabled || trimmed.length === 0) return;
    onSubmit(trimmed);
    setText("");
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    trySubmit();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter submits unless Shift is held. matches the muscle memory of
    // every modern chat UI (Slack, ChatGPT, Claude).
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      trySubmit();
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <TextArea
        aria-label="Question"
        placeholder="用中文提问，比如：桥梁防水谁负责？"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        maxLength={MAX_QUESTION_CHARS}
        rows={2}
        className="flex-1"
      />
      <Button type="submit" disabled={disabled || text.trim().length === 0}>
        Send
      </Button>
    </form>
  );
}
