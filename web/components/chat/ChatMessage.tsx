/**
 * ChatMessage — renders one user or assistant turn.
 *
 * Assistant messages can carry citations + a confidence tag:
 *   - confidence "high" / "medium" → no banner (default trust)
 *   - confidence "low"             → amber "low confidence" banner so
 *                                     the user knows to double-check
 *   - confidence "none"            → blue "retrieval-only" banner
 *                                     (no answerer configured or empty
 *                                     index — see api/routes/ask.py)
 *
 * Layout choice: user messages right-aligned with darker bg, assistant
 * left-aligned with lighter bg. Standard chat affordance, no surprise.
 */

import { CitationChip } from "./CitationChip";
import type { Citation } from "@/lib/api-client";

export interface ChatMessageData {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  confidence?: "high" | "medium" | "low" | "none";
}

interface ChatMessageProps {
  message: ChatMessageData;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  // Chat-bubble styling: user turns are solid blue, right-aligned, with a
  // squared bottom-right corner ("tail"); assistant turns are white with a
  // subtle border + shadow and a squared bottom-left corner. This reads as
  // a conventional chat thread (replaces the earlier flat Card look).
  const bubbleClass = isUser
    ? "max-w-[85%] rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 text-white shadow-sm"
    : "max-w-[85%] rounded-2xl rounded-bl-sm border border-gray-200 bg-white px-4 py-2.5 text-gray-900 shadow-sm";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={bubbleClass}>
        {/* Confidence banner only for non-trivial states. Keeps high/medium
            answers visually clean. English phrase kept (asserted by tests)
            with a Chinese hint appended for the end user. */}
        {message.confidence === "low" && (
          <div className="mb-2 rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800">
            ⚠️ Low confidence · 置信度较低，请核对原始文档
          </div>
        )}
        {message.confidence === "none" && (
          <div className="mb-2 rounded-md bg-blue-50 px-2 py-1 text-xs text-blue-800">
            ℹ️ Retrieval-only mode · 未配置答疑模型，以下为检索命中片段
          </div>
        )}

        <div className="whitespace-pre-wrap text-sm leading-relaxed">
          {message.content}
        </div>

        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.citations.map((c, i) => (
              <CitationChip key={`${c.file}-${c.page}-${i}`} file={c.file} page={c.page} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
