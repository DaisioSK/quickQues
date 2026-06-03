/**
 * Tests for CitationChip.
 *
 * The wire-up contract this chip must respect:
 *   - URL pattern `/pdf/{encodedFile}?page={N}` matches what
 *     ssPdfViewer registers as its dynamic route.
 *   - target=_blank with noopener,noreferrer prevents the opened tab
 *     from reaching back via window.opener.
 *
 * Both are asserted here so a wrong refactor surfaces before reaching
 * the manual ssWire happy-path check.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { CitationChip } from "./CitationChip";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("CitationChip", () => {
  it("renders the file name and page number", () => {
    render(<CitationChip file="Contract DEMO.pdf" page={42} />);
    expect(screen.getByText("Contract DEMO.pdf")).toBeInTheDocument();
    expect(screen.getByText("p.42")).toBeInTheDocument();
  });

  it("opens the pdf viewer URL in a new tab on click", () => {
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);
    render(
      <CitationChip file="Contract DEMO(1of9) TQA.pdf" page={5} />,
    );
    fireEvent.click(screen.getByRole("button"));

    expect(openSpy).toHaveBeenCalledOnce();
    const [url, target, features] = openSpy.mock.calls[0]!;
    expect(url).toBe("/pdf/Contract%20DEMO(1of9)%20TQA.pdf?page=5");
    expect(target).toBe("_blank");
    expect(features).toContain("noopener");
    expect(features).toContain("noreferrer");
  });
});
