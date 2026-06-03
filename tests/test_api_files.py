"""Unit tests for /files/{filename}.

Coverage matches the security model in api/routes/files.py:
  1. happy path — listed PDF returns 200 + application/pdf bytes
  2. unknown filename → 404
  3. path-traversal attempt → 400 (syntactic reject before whitelist)
  4. legitimate PDF outside the whitelist → 404 (placed on disk but
     not in os.listdir at request time — tests the dir-scoped probe)

Strategy: monkeypatch `INPUT_DOCS_DIR` to a tmp dir so the test is
hermetic. No real input-docs/ access; no Qdrant; no Anthropic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jcontract.api.main import create_app
from jcontract.api.routes import files as files_route

# Minimal PDF header. Three bytes (`%PDF`) is enough for content sniffing,
# but we use a slightly fuller fixture so byte-length asserts are meaningful.
_MINIMAL_PDF_BYTES = b"%PDF-1.4\n%MockPDFForTests\n%%EOF\n"


@pytest.fixture
def tmp_input_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect INPUT_DOCS_DIR to a fresh tmp dir for each test.

    Yields the dir Path so tests can drop fixture PDFs in it before
    making requests. Cleanup is automatic via the tmp_path fixture.
    """
    docs = tmp_path / "input-docs"
    docs.mkdir()
    monkeypatch.setattr(files_route, "INPUT_DOCS_DIR", docs)
    return docs


def _client() -> TestClient:
    """Build a TestClient against a fresh app (no shared state)."""
    return TestClient(create_app())


def test_serves_whitelisted_pdf_with_correct_content_type(tmp_input_docs: Path) -> None:
    """Happy path: file in input-docs/ → 200 + application/pdf."""
    pdf = tmp_input_docs / "sample.pdf"
    pdf.write_bytes(_MINIMAL_PDF_BYTES)

    resp = _client().get("/files/sample.pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content == _MINIMAL_PDF_BYTES


def test_returns_404_for_filename_not_in_whitelist(tmp_input_docs: Path) -> None:
    """File never ingested → 404, no info leak about other files."""
    # Drop a real PDF in the dir to ensure the route isn't ALWAYS 404ing
    # — the failing path is specifically "unknown filename".
    (tmp_input_docs / "real.pdf").write_bytes(_MINIMAL_PDF_BYTES)

    resp = _client().get("/files/does-not-exist.pdf")

    assert resp.status_code == 404


def test_rejects_path_traversal_attempts(tmp_input_docs: Path) -> None:
    """Path-traversal attacks are denied across multiple defense layers.

    Defense-in-depth means a given attack may surface at either:
      - the HTTP layer (httpx + Starlette normalise `/files/..` → `/files`
        before our route sees it, yielding 404 because no `/files` route
        exists),
      - or our route's `_is_safe_filename` guard (when the encoded form
        survives normalization, e.g. `..%2Fetc%2Fpasswd` decodes to
        `../etc/passwd` and hits the slash check → 400).

    Both outcomes are secure. The test asserts the attacker NEVER gets
    a 200 + a file body, not the specific status code (which depends on
    where in the stack the rejection lands).
    """
    client = _client()

    # URL-encoded slash → httpx decodes → our route sees "../etc/passwd"
    # → _is_safe_filename rejects "/" → 400.
    resp_slash_encoded = client.get("/files/..%2Fetc%2Fpasswd")
    assert resp_slash_encoded.status_code == 400

    # Plain `..` — httpx normalises `/files/..` to `/files`, no such
    # route → 404. Our guard never gets called but security holds.
    resp_dotdot = client.get("/files/..")
    assert resp_dotdot.status_code in {400, 404}
    assert resp_dotdot.status_code != 200


def test_legitimate_pdf_outside_dir_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symlink / file living elsewhere → not whitelisted by listdir → 404.

    Scenario: a PDF exists on disk just outside input-docs/ but a
    misconfigured client requests it by name. Our listdir whitelist
    must reject — even though the file is real, it's not where it
    needs to be.
    """
    # Real PDF placed OUTSIDE the configured input-docs/.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "stolen.pdf").write_bytes(_MINIMAL_PDF_BYTES)

    # Configure input-docs/ to an EMPTY dir.
    empty_docs = tmp_path / "input-docs"
    empty_docs.mkdir()
    monkeypatch.setattr(files_route, "INPUT_DOCS_DIR", empty_docs)

    resp = _client().get("/files/stolen.pdf")

    # listdir(input-docs/) = [] → not in whitelist → 404
    assert resp.status_code == 404
