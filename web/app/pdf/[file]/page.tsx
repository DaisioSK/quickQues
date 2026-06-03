/**
 * /pdf/[file]?page=N — PDF viewer page (MVP step 4/5).
 *
 * Architecture: Server Component renders a full-height iframe pointing
 * at the backend `/files/{filename}#page=N` endpoint. The browser's
 * built-in PDF viewer renders the bytes and honours the `#page=N`
 * fragment to jump to the cited page.
 *
 * Why iframe + native viewer over react-pdf (DECISION-mvp.pdf.3):
 *   - Zero new frontend dependency (~50KB JS + pdf.js worker avoided)
 *   - Native viewer = fastest rendering, best quality, free range-loading
 *   - `#page=N` is a portable spec (Adobe PDF Open Parameters) honoured
 *     by Chrome / Edge / Firefox / Safari built-ins
 *   - MVP scope (DECISION-orch-6) is page-level jump only — perfect fit
 *
 * Route shape per Next.js 16 App Router conventions:
 *   - `params` and `searchParams` are Promises (since Next 15) — must
 *     be awaited in this async Server Component.
 *   - `params.file` arrives STILL percent-encoded in this Next version
 *     (contrary to older Next, which decoded it — see web/AGENTS.md).
 *     CitationChip encodeURIComponent's the filename before navigation,
 *     so we must decodeURIComponent it back to the raw filename here
 *     before handing it to pdfFileUrl (which re-encodes). Without the
 *     decode, a filename with spaces/parens like
 *     "Contract DEMO(1of9) TQA.pdf" gets double-encoded (%20 → %2520)
 *     and the backend whitelist lookup 404s.
 */

import Link from "next/link";
import { pdfFileUrl } from "@/lib/api-client";

type Params = Promise<{ file: string }>;
type SearchParams = Promise<{ page?: string }>;

interface PdfPageProps {
  params: Params;
  searchParams: SearchParams;
}

function parsePage(raw: string | undefined): number | undefined {
  // No param → undefined (pdfFileUrl drops the fragment).
  if (!raw) return undefined;
  const n = Number.parseInt(raw, 10);
  // NaN / non-positive → also drop. PDF pages are 1-indexed; out-of-
  // range high numbers harmlessly clip to the last page in the viewer.
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

export default async function PdfPage({ params, searchParams }: PdfPageProps) {
  const { file: rawFile } = await params;
  const sp = await searchParams;
  const page = parsePage(sp.page);

  // Next hands us the percent-encoded segment; decode to the real
  // filename before pdfFileUrl re-encodes it. Guard against a malformed
  // sequence by falling back to the raw value.
  let file: string;
  try {
    file = decodeURIComponent(rawFile);
  } catch {
    file = rawFile;
  }

  const src = pdfFileUrl(file, page);

  return (
    <div className="flex h-screen flex-col bg-gray-900">
      <header className="flex items-center justify-between border-b border-gray-700 bg-gray-800 px-4 py-2 text-sm text-gray-100">
        <div className="flex items-center gap-3 truncate">
          <span className="font-medium">{file}</span>
          {page !== undefined && (
            <span className="rounded bg-gray-700 px-2 py-0.5 text-xs">
              page {page}
            </span>
          )}
        </div>
        <Link
          href="/"
          className="text-xs text-gray-300 hover:text-white"
        >
          ← Back to chat
        </Link>
      </header>

      <iframe
        src={src}
        title={`PDF viewer: ${file}`}
        className="flex-1 w-full border-0"
      />
    </div>
  );
}
