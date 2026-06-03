"""PDFParser Protocol — Layer 0.

Default impl: impls/pypdf_parser.py (Phase 1 S1.1 ssA).
Replacement candidates per docs/project_guideline.md §4:
  - LlamaParse (cloud, better for tables/drawings, costs ~$3/1000 pages)
  - MinerU (local, GPU-accelerated, better for scanned PDFs)
  - pdfplumber (alt local, sometimes better at columns)

Business code must depend on this Protocol, never on a concrete impl.
The impl is wired via config.yaml at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .schema import ParsedPage


class PDFParser(Protocol):
    """Convert a PDF file into ordered per-page text + table extracts.

    Implementations should:
      - Open the file at ``pdf_path`` and read every page in document order.
      - Return one ParsedPage per source page with 1-indexed ``page_num``.
      - Never raise on extraction-quality issues for a single page; emit
        an empty ``text`` and let the chunker filter.
      - Raise on file-level errors (not found, corrupt PDF header) so the
        ingest pipeline can fail loudly.
    """

    def parse(self, pdf_path: Path) -> list[ParsedPage]: ...
