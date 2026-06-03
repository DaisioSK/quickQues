/**
 * SuggestedQuestions — example prompts shown in the empty chat state.
 *
 * Why this exists (user feedback 2026-05-30): a non-domain-expert user
 * "doesn't know whether they're asking the right thing". Seeding a few
 * concrete, verified-to-answer questions turns the blank page into a
 * one-click starting point and teaches the kind of question the corpus
 * can answer.
 *
 * The system is domain-agnostic (Phase 7); the DEFAULT_SUGGESTIONS below
 * are the construction (DEMO) example set, kept because contract is the
 * default deployment corpus. A different deployment passes its own via
 * the optional `questions` prop; when Web upload (Enhancement E4) lands
 * these should come from the active collection's DomainProfile.
 */

interface SuggestedQuestionsProps {
  /** Called with the chosen question text; parent submits it as a turn. */
  onPick: (question: string) => void;
  /** Disable while a request is in flight. */
  disabled?: boolean;
  /** Example questions to show; defaults to the DEMO construction set. */
  questions?: readonly string[];
}

export const DEFAULT_SUGGESTIONS: readonly string[] = [
  "谁来负责这座桥梁的建造？",
  "TSA 的 platform level 有什么特殊要求？",
  "M&E raiser 有什么要求？",
  "tender clarification 是什么时候交的？谁交的？",
  "DES 屋顶有什么要求？",
];

export function SuggestedQuestions({
  onPick,
  disabled = false,
  questions = DEFAULT_SUGGESTIONS,
}: SuggestedQuestionsProps) {
  return (
    <div className="mx-auto mt-16 max-w-xl text-center">
      <div className="mb-1 text-3xl">📄💬</div>
      <h2 className="text-base font-semibold text-gray-800">
        开始向已索引的文档提问
      </h2>
      <p className="mt-1 text-sm text-gray-500">
        用中文提问即可，点击下面的示例试试看：
      </p>
      <div className="mt-5 flex flex-wrap justify-center gap-2">
        {questions.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onPick(q)}
            className="rounded-full border border-gray-200 bg-white px-4 py-2 text-sm text-gray-700 shadow-sm transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
