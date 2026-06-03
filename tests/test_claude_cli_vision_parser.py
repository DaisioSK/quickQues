"""Unit tests for ClaudeCliVisionParser — focused on E10 model-aware caching.

No subprocess is ever spawned: we inject a fake ``claude_path`` so the
binary-presence check passes at __init__, and monkeypatch
``_call_claude_cli`` so the parse path never shells out. The pypdfium2
render of the real synthetic fixture IS exercised (local, fast, free).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jcontract.impls.claude_cli_vision_parser import DEFAULT_MODEL, ClaudeCliVisionParser

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")
# Any existing executable satisfies shutil.which-style resolution in
# __init__; the parser never actually runs it because we patch the call.
FAKE_CLAUDE = "/bin/true"


def _parser(
    tmp_path: Path,
    *,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    ocr_text: str = "page text",
) -> ClaudeCliVisionParser:
    p = ClaudeCliVisionParser(
        cache_dir=tmp_path / "cache",
        render_dir=tmp_path / "render",
        model=model,
        max_pages=1,
        claude_path=FAKE_CLAUDE,
    )
    monkeypatch.setattr(p, "_call_claude_cli", lambda *a, **k: ocr_text)
    return p


def test_default_model_writes_legacy_unsuffixed_cache(tmp_path, monkeypatch):
    """Default 'haiku' must keep the pre-E10 `<hash>.text.txt` filename so
    the maintainer's existing full-ingest cache still hits."""
    assert DEFAULT_MODEL == "haiku"  # guards the backward-compat contract
    parser = _parser(tmp_path, model="haiku", monkeypatch=monkeypatch)
    parser.parse(SYNTHETIC_PDF)

    cache_files = list((tmp_path / "cache").glob("*.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].name.endswith(".text.txt")
    assert ".haiku." not in cache_files[0].name


def test_non_default_model_writes_suffixed_cache(tmp_path, monkeypatch):
    """A non-default model lands in its own namespace (`.text.sonnet.txt`)."""
    parser = _parser(tmp_path, model="sonnet", monkeypatch=monkeypatch)
    parser.parse(SYNTHETIC_PDF)

    cache_files = list((tmp_path / "cache").glob("*.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].name.endswith(".text.sonnet.txt")


def test_haiku_and_sonnet_do_not_share_cache(tmp_path, monkeypatch):
    """Same page + same cache dir, different model → two distinct cache
    files (the bug E10 fixes: sonnet must not return cached haiku text)."""
    haiku = _parser(tmp_path, model="haiku", monkeypatch=monkeypatch, ocr_text="HAIKU TEXT")
    haiku.parse(SYNTHETIC_PDF)
    sonnet = _parser(tmp_path, model="sonnet", monkeypatch=monkeypatch, ocr_text="SONNET TEXT")
    pages = sonnet.parse(SYNTHETIC_PDF)

    cache_files = sorted(f.name for f in (tmp_path / "cache").glob("*.txt"))
    assert len(cache_files) == 2  # not a collision
    # The sonnet parser produced its own fresh OCR, not the haiku cache hit.
    assert pages[0].text == "SONNET TEXT"
