/**
 * Chat home — the entire MVP front end lives here for step 3/5.
 *
 * Per DECISION-mvp.chat.1 chat is at `/` rather than `/chat`. Layout:
 *   header (brand + retrieval-mode hint)
 *   ↓
 *   scrollable message list (empty state = SuggestedQuestions)
 *   ↓
 *   fixed input bar at bottom
 *
 * State machine (kept tiny on purpose — no Redux/Zustand per DECISION
 * deferred to Enhancement):
 *   messages: list of user + assistant turns
 *   loading:  true while a question is in flight
 *   error:    last error string from the API client; null when clear
 *
 * Why no SSE / streaming: DECISION-mvp.api.1 — backend returns a single
 * JSON payload; we render once the promise resolves. Users see an
 * animated "Thinking…" placeholder during the 5-10s wait. Streaming is
 * in the Enhancement queue (FORESHADOW-mvp.api.1).
 *
 * Visual polish + suggested questions added 2026-05-30 per user feedback
 * (UI "有点丑" + non-expert "不知道问得对不对").
 */

"use client";

import { useEffect, useRef, useState } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessage, type ChatMessageData } from "@/components/chat/ChatMessage";
import { SuggestedQuestions } from "@/components/chat/SuggestedQuestions";
import { askQuestion } from "@/lib/api-client";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Auto-scroll anchor: keep the newest turn / "Thinking…" in view.
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Optional-chain the method too: jsdom (test env) doesn't implement
    // scrollIntoView, and we don't want that to throw in the effect.
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(question: string) {
    // Optimistically append the user turn so the input feels responsive.
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);
    setError(null);

    try {
      const response = await askQuestion(question);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations,
          confidence: response.confidence,
        },
      ]);
    } catch (err) {
      // We surface the error via a small banner rather than as an
      // assistant message — keeps the message list pure (only successful
      // turns) and makes retry semantics clearer.
      const detail = err instanceof Error ? err.message : String(err);
      setError(detail);
    } finally {
      setLoading(false);
    }
  }

  const isEmpty = messages.length === 0 && !loading;

  return (
    <div className="flex h-screen flex-col bg-gradient-to-b from-gray-50 to-gray-100">
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/80 px-6 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-sm font-bold text-white">
            J
          </span>
          <div>
            <h1 className="text-base font-semibold leading-tight text-gray-900">
              j-contract
            </h1>
            <p className="text-xs text-gray-500">
              中文提问 · 点击引用跳转到 PDF 对应页
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {isEmpty && <SuggestedQuestions onPick={handleSubmit} disabled={loading} />}

          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} />
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm border border-gray-200 bg-white px-4 py-3 shadow-sm">
                {/* sr-only label keeps the loading state announced to screen
                    readers (and asserted by tests) while the dots animate. */}
                <span className="sr-only">Thinking…</span>
                <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400" />
              </div>
            </div>
          )}

          {error && (
            <div
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
            >
              出错了：{error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-gray-200 bg-white px-6 py-3">
        <div className="mx-auto max-w-3xl">
          <ChatInput onSubmit={handleSubmit} disabled={loading} />
          <p className="mt-2 text-center text-[11px] text-gray-400">
            答案由 AI 基于已索引文档生成，请以原始 PDF 文档为准。
          </p>
        </div>
      </footer>
    </div>
  );
}
