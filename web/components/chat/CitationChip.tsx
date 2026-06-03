/**
 * CitationChip — clickable badge linking to a specific PDF page.
 *
 * Rendered in assistant ChatMessage's citation row. Click opens the
 * ssPdfViewer route in a new tab per DECISION-mvp.chat.2 ("user must
 * not lose chat context when inspecting a citation").
 *
 * URL contract: `/pdf/{encodedFile}?page={N}` — this MUST match the
 * dynamic route that ssPdfViewer (MVP step 4/5) will register at
 * `web/app/pdf/[file]/page.tsx`. If either side moves, the wire-up
 * sub-sprint (ssWire) is responsible for re-aligning them.
 *
 * Why not <a target="_blank">: identical behaviour visually, but using
 * button + window.open keeps the keyboard/screen-reader semantics
 * "button that does something" rather than "navigation link" — more
 * accurate for chip-style affordances. Also lets us spy/test the call
 * cleanly without jsdom navigation gotchas.
 */

interface CitationChipProps {
  file: string;
  page: number;
}

export function CitationChip({ file, page }: CitationChipProps) {
  function handleClick() {
    const url = `/pdf/${encodeURIComponent(file)}?page=${page}`;
    // `noopener,noreferrer` so the opened tab can't reach back to this
    // window via window.opener — defense in depth even though we
    // control the destination.
    window.open(url, "_blank", "noopener,noreferrer");
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700 transition-colors hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
      title={`Open ${file} at page ${page}`}
    >
      <span className="truncate max-w-[200px]">{file}</span>
      <span className="text-blue-400">·</span>
      <span className="whitespace-nowrap">p.{page}</span>
      {/* open-in-new affordance — signals the chip navigates to the PDF */}
      <span aria-hidden className="text-blue-400">↗</span>
    </button>
  );
}
