"""pypdf-based PDFParser implementation.

What:
    Reads a PDF page-by-page using the pure-Python ``pypdf`` library and
    returns ``ParsedPage`` records with 1-indexed page numbers.

Why pypdf (DECISION):
    pypdf is the lightest text-extractor we ship with (no system deps, no
    GPU, no network). It's the Phase 1 default per the interfaces/parser.py
    docstring. If extraction quality on a given PDF is poor (e.g. the DEMO
    samples are scanned image-only PDFs), we degrade gracefully: emit empty
    ``ParsedPage.text`` and let downstream OCR (FORESHADOW Phase 2) fill
    in. We do NOT fall back to pdfplumber here — the interface allows
    swapping the whole parser via config when a real upgrade is needed.

Context:
    Phase 1 S1.1 ssA. Consumed by ingest/pipeline.py (written by the
    integrator). The chunker (qa_chunker.py) takes our output as input.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from jcontract.interfaces.schema import ParsedPage


class PyPdfParser:
    """Concrete ``PDFParser`` backed by ``pypdf``.

    Stateless: a single instance can be reused across files. Each
    ``parse()`` call opens the file fresh — we do not keep file handles.
    """

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        """Extract per-page text from ``pdf_path``.

        Contract (from interfaces/parser.py):
          - 1-indexed page_num
          - Never raise on single-page extraction errors; emit empty text
          - DO raise on file-level errors (missing file, corrupt header)

        Implementation notes:
          - ``PdfReader`` raises on file-not-found / unreadable header,
            which we propagate (loud failure at ingest time).
          - ``page.extract_text()`` can return ``None`` for image-only or
            malformed pages — we coerce to ``""`` so downstream code never
            needs to None-check.
          - ``tables`` stays empty: pypdf does not isolate tables. A
            future LlamaParse impl can populate this field.
        """
        # Let PdfReader raise on file-level errors — caller (ingest pipeline)
        # needs to see them, per Hard Rule "fail loudly".
        reader = PdfReader(str(pdf_path))

        pages: list[ParsedPage] = []
        for idx, page in enumerate(reader.pages):
            # Per-page extraction may fail on weird PDFs; we swallow only
            # extraction-quality errors (the parser contract permits this)
            # and emit empty text so the chunker can still iterate.
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001  Why: contract requires never
                # raising on per-page issues; only file-level failures
                # bubble. We capture the empty result and continue.
                text = ""

            pages.append(
                ParsedPage(
                    page_num=idx + 1,  # 1-indexed for UI/citation alignment
                    text=text,
                    tables=[],
                )
            )

        return pages
