/**
 * Typed HTTP client for the j-contract FastAPI backend.
 *
 * Shape contract mirrors src/jcontract/api/schemas.py:
 *   AskRequest  -> AskRequest below
 *   AskResponse -> AskResponse below
 *   CitationOut -> Citation below
 *
 * Why we duplicate Pydantic types in TS instead of generating from
 * OpenAPI: at MVP scale (3 endpoints, ~6 fields total) hand-writing is
 * faster than wiring an openapi-typescript generator + verifying its
 * output. Enhancement queue can add generation when surface grows.
 *
 * Why this module lives in ssNextInit (prep) and not ssChat: per
 * DECISION-mvp.next.2 — keeping the fetch wrapper here means ssChat
 * and ssPdfViewer don't have to coordinate on HTTP plumbing. ssChat
 * only appends `askQuestion` at the end of this file (or its own
 * caller-side wrapper); ssPdfViewer uses the PDF iframe directly
 * with the backend URL, no fetch needed.
 */

// Backend base URL. Hard-coded for MVP since CORS in api/main.py only
// allows localhost:3000 anyway (DECISION-mvp.api.1 scope). Lifted to
// env var when we land docker-compose (Enhancement E6).
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/** Matches Pydantic AskRequest in src/jcontract/api/schemas.py */
export interface AskRequest {
  question: string;
}

/** Matches Pydantic CitationOut */
export interface Citation {
  file: string;
  page: number;
}

/**
 * Matches Pydantic AskResponse. `confidence: "none"` indicates the
 * backend ran in retrieval-only mode (no answerer configured) or hit
 * the empty-index fallback. The UI should render a banner in those
 * cases per the api/routes/ask.py graceful-degradation contract.
 */
export interface AskResponse {
  answer: string;
  citations: Citation[];
  confidence: "high" | "medium" | "low" | "none";
}

/** Matches Pydantic SearchResultOut (debug endpoint). */
export interface SearchResult {
  file: string;
  page: number;
  chunk_type: string;
  score: number;
  preview: string;
}

/**
 * Generic JSON fetch wrapper.
 *
 * Throws on non-2xx with the response body in the error message so
 * callers can show a useful "backend said: ..." message rather than
 * "fetch failed". Network errors propagate the original TypeError.
 *
 * Why no retry / backoff: MVP. The user can hit "submit" again. Add
 * a retry policy when the surface stops being a single chat box.
 */
export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const resp = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!resp.ok) {
    // We try to read the body for diagnostic surface, but never throw
    // if the body itself is malformed — bad payload on a bad status
    // shouldn't hide the status code.
    let body = "";
    try {
      body = await resp.text();
    } catch {
      body = "(body unreadable)";
    }
    throw new Error(`API ${resp.status} ${resp.statusText}: ${body.slice(0, 500)}`);
  }

  return (await resp.json()) as T;
}

/**
 * Submit a question to the backend.
 *
 * This is the one method ssChat needs; it lives here (in the prep ss)
 * so ssChat's component code stays focused on UI rather than HTTP.
 */
export function askQuestion(question: string): Promise<AskResponse> {
  return fetchJson<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify({ question } satisfies AskRequest),
  });
}

/**
 * Build a URL pointing at the backend's PDF stream endpoint.
 *
 * Used by ssPdfViewer's iframe src. Kept here so the URL pattern lives
 * next to the rest of the backend contract — if the backend route
 * moves we change it in one place.
 *
 * Optional `page` appends a `#page=N` fragment. PDF viewers (Chrome,
 * Edge, Firefox, Safari built-in) all honour this fragment per the
 * Adobe PDF Open Parameters spec — jumping to a specific page without
 * any JS. DECISION-mvp.pdf.3 in docs/dev-sprint.md (iframe over
 * react-pdf for MVP).
 */
export function pdfFileUrl(filename: string, page?: number): string {
  const base = `${API_BASE}/files/${encodeURIComponent(filename)}`;
  return page && page > 0 ? `${base}#page=${page}` : base;
}
