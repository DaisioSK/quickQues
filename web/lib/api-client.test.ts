/**
 * Tests for the api-client.ts fetch wrapper.
 *
 * Strategy: stub global.fetch via vi.spyOn — no network calls, just
 * assert the wrapper builds the right URL/body and parses responses
 * the way ssChat will consume them.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  askQuestion,
  fetchJson,
  pdfFileUrl,
  type AskResponse,
} from "./api-client";

afterEach(() => {
  vi.restoreAllMocks();
});

function mockFetchOk(body: unknown): void {
  vi.spyOn(global, "fetch").mockResolvedValue(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );
}

describe("fetchJson", () => {
  it("posts JSON and returns the parsed body", async () => {
    const spy = vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );

    const result = await fetchJson<{ ok: boolean }>("/test", {
      method: "POST",
      body: JSON.stringify({ a: 1 }),
    });

    expect(result).toEqual({ ok: true });
    // Wrapper prepended the API_BASE constant; we just assert the path suffix
    // to avoid coupling the test to env-dependent base URL.
    const calledUrl = spy.mock.calls[0]?.[0];
    expect(typeof calledUrl).toBe("string");
    expect(String(calledUrl)).toMatch(/\/test$/);
  });

  it("throws on non-2xx with body included in the error message", async () => {
    // Each fetchJson() call consumes the Response body via .text(), so
    // we mock fetch to return a FRESH Response per call rather than
    // sharing one (the second .text() on a consumed Response is empty
    // and the assertion below would mask the actual behaviour).
    vi.spyOn(global, "fetch").mockImplementation(
      async () =>
        new Response("server exploded", { status: 500, statusText: "Server Error" }),
    );

    await expect(fetchJson("/boom")).rejects.toThrow(/500/);
    await expect(fetchJson("/boom")).rejects.toThrow(/server exploded/);
  });
});

describe("askQuestion", () => {
  it("POSTs the question and decodes AskResponse", async () => {
    const expected: AskResponse = {
      answer: "中文答案",
      citations: [{ file: "f.pdf", page: 1 }],
      confidence: "high",
    };
    mockFetchOk(expected);

    const got = await askQuestion("桥梁防水谁负责?");
    expect(got).toEqual(expected);
  });
});

describe("pdfFileUrl", () => {
  it("encodes the filename and includes the /files/ path", () => {
    const url = pdfFileUrl("Contract DEMO(1of9) TQA.pdf");
    expect(url).toMatch(/\/files\//);
    // Space + parens must be URL-encoded so the resulting URL is valid.
    expect(url).toContain(encodeURIComponent("Contract DEMO(1of9) TQA.pdf"));
    // No page param → no fragment.
    expect(url).not.toContain("#page=");
  });

  it("appends a #page=N fragment when page is provided", () => {
    const url = pdfFileUrl("doc.pdf", 7);
    expect(url).toMatch(/#page=7$/);
  });

  it("omits the fragment when page is 0 or negative (defensive)", () => {
    // Treat 0 / negative as "no page specified" — guards against the
    // caller passing untrusted numbers without bounds-checking.
    expect(pdfFileUrl("doc.pdf", 0)).not.toContain("#page=");
    expect(pdfFileUrl("doc.pdf", -1)).not.toContain("#page=");
  });
});
