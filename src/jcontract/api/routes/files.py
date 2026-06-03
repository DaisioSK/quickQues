"""`/files/{filename}` — stream a PDF from the ingest directory.

This is the backend endpoint the ssPdfViewer frontend points its iframe
at via api-client.pdfFileUrl(). The browser's native PDF viewer renders
the bytes; we set the right Content-Type so it inlines (not downloads)
and we honour the `#page=N` URL fragment that the chat citation chip
appends (the spec is handled entirely browser-side; the backend just
serves the bytes).

Security model (DECISION-mvp.pdf.1 in docs/dev-sprint.md):

- Whitelist by `os.listdir("input-docs/")`: only files present at
  process start are servable. MVP has no Web upload (DECISION-orch-7),
  so the whitelist is effectively static. Adding a PDF means
  `jcontract ingest <pdf>` (which copies to input-docs/) + a backend
  restart so listdir picks it up.

- Reject filenames with path separators or `..` BEFORE the whitelist
  check. Belt-and-braces — the whitelist alone would already block
  these because no such entry exists in listdir output, but doing the
  syntactic check first means audit logs surface "obvious attack" vs
  "honest typo" cleanly.

- Final defense: `Path.resolve()` + `is_relative_to(input_docs.resolve())`
  in case a future refactor accidentally removes one of the earlier
  guards.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = structlog.get_logger(__name__)

router = APIRouter()


# Single source of truth for where we look up PDFs. Lifted into a module
# constant so tests can monkeypatch a tmp dir without touching prod state.
INPUT_DOCS_DIR = Path("input-docs")


def _is_safe_filename(name: str) -> bool:
    """Reject anything that's clearly trying to escape the directory.

    A safe filename is a single path segment with no separators and no
    parent-directory markers. We don't try to enumerate exhaustive
    encoded-traversal variants here — the whitelist check downstream is
    the real gate, this is just an obvious-attack fast-fail.
    """
    if not name:
        return False
    if "/" in name or "\\" in name:
        return False
    if name == ".." or name.startswith("../"):
        return False
    # Reject null bytes; some platforms truncate at \x00 which can
    # bypass extension checks.
    return "\x00" not in name


@router.get("/files/{filename:path}")
def get_file(filename: str) -> FileResponse:
    """Stream the requested PDF if it's in the ingest-dir whitelist."""

    # Syntactic guard first — `{filename:path}` accepts slashes, so we
    # must reject them explicitly. A legitimate request never contains
    # them.
    if not _is_safe_filename(filename):
        logger.warning("api.files.unsafe_filename", filename=filename[:120])
        raise HTTPException(status_code=400, detail="invalid filename")

    if not INPUT_DOCS_DIR.exists():
        # Configuration issue (input-docs/ missing). 404 not 500 because
        # this is "no files servable" from the client's perspective.
        logger.error("api.files.input_docs_missing", path=str(INPUT_DOCS_DIR))
        raise HTTPException(status_code=404, detail="no documents available")

    # Whitelist via listdir. Re-read on every request rather than caching
    # so a newly-ingested PDF becomes servable without restart, matching
    # the maintainer's mental model ("I ran `jcontract ingest`, the file
    # should be reachable now").
    available = set(os.listdir(INPUT_DOCS_DIR))
    if filename not in available:
        logger.info("api.files.not_in_whitelist", filename=filename[:120])
        raise HTTPException(status_code=404, detail=f"file not found in {INPUT_DOCS_DIR}/")

    file_path = INPUT_DOCS_DIR / filename

    # Belt-and-braces resolve check. If filename somehow contained
    # encoded segment separators that slipped past `_is_safe_filename`,
    # `.resolve()` would escape input-docs and this check fires.
    resolved = file_path.resolve()
    if not resolved.is_relative_to(INPUT_DOCS_DIR.resolve()):
        logger.warning("api.files.path_escape_attempt", filename=filename[:120])
        raise HTTPException(status_code=403, detail="path escapes ingest dir")

    if not resolved.is_file():
        # listdir said yes but it's not a regular file (could be a dir,
        # symlink to nowhere, etc.). 404 still — same client-side meaning.
        raise HTTPException(status_code=404, detail="not a regular file")

    logger.info("api.files.serve", filename=filename, bytes=resolved.stat().st_size)
    # FileResponse handles range requests for free, which the browser's
    # PDF viewer uses to fetch only the page it needs to render first.
    return FileResponse(
        path=str(resolved),
        media_type="application/pdf",
        # `filename` here drives the Content-Disposition header; the
        # browser uses it for "Save as" if the user downloads. We feed
        # back the safe whitelisted name (NOT a user-supplied alias).
        filename=filename,
    )
