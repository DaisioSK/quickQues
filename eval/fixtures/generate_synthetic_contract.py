"""Generate a synthetic text-based PDF mimicking the DEMO TQA structure.

Why this exists:
- The real input-docs/Contract DEMO(1of9) TQA.pdf is 100% image-only scans
  (verified by inspecting PDF content streams — only image-show ops). pypdf
  extracts 0 text from it.
- Adding OCR (PaddleOCR or Claude Vision) is a separate sub-sprint
  (FORESHADOW Phase 1.5 in dev_log).
- For the Phase 1 prototype, we still need to demonstrate the full pipeline
  end-to-end: parse -> chunk -> embed -> index -> search -> eval. This
  generator produces a small text-based PDF that mirrors realistic DEMO
  Tender Q&A structure: Section/Clause headers, Question No. / Answer
  blocks, Drawing No. cross-references, Revision markers.

The synthetic fixture lives at eval/fixtures/synthetic_contract_tqa.pdf and is
gitignored along with other data/ artifacts (see .gitignore data/* rule —
fixtures live under eval/fixtures/ which we keep tracked, but the .pdf
output itself can be re-generated and shouldn't bloat the repo).

Run: `python eval/fixtures/generate_synthetic_contract.py`
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUTPUT_PATH = Path(__file__).parent / "synthetic_contract_tqa.pdf"


# Content modeled on the WeChat screenshot the user provided (TQA ACME/
# TRACKWORK/16 about waterproofing at pier), plus plausible siblings for
# the other golden-case categories.
PAGES: list[dict[str, str]] = [
    {
        "title": "Section 1 - Project Overview",
        "body": (
            "PROJECT DEMO - TENDER QUESTION AND ANSWER (TQA)\n"
            "\n"
            "Name of Tenderer: ACME DESIGN AND CONSTRUCTION PTE. LTD.\n"
            "Date of issue: 8 Nov 2019\n"
            "\n"
            "This document consolidates all Tender Clarifications raised by\n"
            "the Tenderer and the corresponding answers issued by the\n"
            "Authority during the tender period.\n"
            "\n"
            "Section 1 defines key terminology used throughout the contract.\n"
            "TSA refers to Temporary Staging Area, a designated zone on the\n"
            "construction site where materials, equipment, and contractor\n"
            "facilities may be located during the works. The total TSA\n"
            "allocated to the DEMO project consists of two parcels: a primary\n"
            "TSA of 0.86 hectares and a secondary TSA of 0.25 hectares,\n"
            "providing a total of 1.11 hectares of temporary staging area.\n"
            "\n"
            "Refer to Drawing No. T/PRJ/CWD/WS/2101A for the spatial layout\n"
            "of both TSA parcels and Clause 7 of the Particular Specification\n"
            "for the conditions of use.\n"
        ),
    },
    {
        "title": "Section 7 - Civil and Trackwork Interface",
        "body": (
            "Question No.: ACME/TRACKWORK/16\n"
            "Date of issue: 8 Nov 2019\n"
            "Subject: Civil and Trackwork Interface\n"
            "\n"
            "Question: Refer to Drawing 201436-BEN-T-1180 Rev A interface\n"
            "details between viaduct beams at pier, the screed to\n"
            "waterproofing is stated as to be provided by Trackwork\n"
            "Contractor. This deviates from the Authority's Drawing No.\n"
            "T/PRJ/CWD/CV/7021/A. Please confirm that you will follow the\n"
            "arrangement shown on Authority's Drawing.\n"
            "\n"
            "Answer (Revision 0):\n"
            "We confirm. The screed to waterproofing at the pier shall be\n"
            "provided by the Trackwork Contractor in accordance with\n"
            "Authority Drawing No. T/PRJ/CWD/CV/7021/A and Clause 7.3 of\n"
            "the Particular Specification. The Tenderer's interface drawing\n"
            "shall be revised to align with the Authority's drawing prior\n"
            "to contract execution.\n"
        ),
    },
    {
        "title": "Section 7 - Tender Clarification Procedure",
        "body": (
            "Question No.: ACME/PROC/04\n"
            "Subject: Tender Clarification Submission Procedure\n"
            "\n"
            "Question: Please clarify the procedure for submitting tender\n"
            "clarifications and the expected turnaround time for responses.\n"
            "\n"
            "Answer (Revision 0):\n"
            "Tender Clarifications shall be submitted in writing to the\n"
            "Authority's representative via email, with a copy to the\n"
            "designated project mailbox. Each clarification shall reference\n"
            "the relevant Section, Clause, and Drawing No. where applicable.\n"
            "The Authority shall endeavour to respond within ten (10) working\n"
            "days of receipt. Responses may be issued as Revision 0 (initial)\n"
            "or subsequent revisions (Rev A, Rev B, etc.) where additional\n"
            "clarification is needed.\n"
        ),
    },
    {
        "title": "Section 7 - Civil and Trackwork Interface (cont.)",
        "body": (
            "Question No.: ACME/TRACKWORK/16 (Rev A)\n"
            "\n"
            "Revised Answer (Rev A):\n"
            "Further to Revision 0, the Trackwork Contractor's scope is\n"
            "expanded to include not only the screed to waterproofing at\n"
            "viaduct beam piers but also the corresponding inspection and\n"
            "snagging works prior to handover. Refer to Drawing No.\n"
            "T/PRJ/CWD/CV/7021/A for the updated interface details. Clause\n"
            "7.3 of the Particular Specification governs the responsibility\n"
            "split. This Rev A supersedes Revision 0 in its entirety.\n"
            "\n"
            "Section 7 references: Clause 7.1 (Scope), Clause 7.3 (Civil/\n"
            "Trackwork interface responsibility), Clause 7.5 (Acceptance\n"
            "criteria), and Drawing T/PRJ/CWD/WS/2101A (site layout).\n"
        ),
    },
]


def build_pdf(output: Path) -> None:
    """Render PAGES into a multi-page PDF using fpdf2's core fonts."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    for page in PAGES:
        pdf.add_page()
        # Header — bold title at top
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.multi_cell(0, 7, page["title"])
        pdf.ln(2)
        # Body
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, page["body"])

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))


if __name__ == "__main__":
    build_pdf(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Pages: {len(PAGES)}")
