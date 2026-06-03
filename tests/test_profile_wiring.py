"""Phase 7 SS4: parsers + captioner take prompts from the active DomainProfile.

Verifies the `document` profile (a) sends its neutral prompt to the model and
(b) writes to a separate, profile-suffixed cache namespace — so re-OCRing the
same bytes under a different domain re-runs instead of returning contract's output.
No real API: mocked clients only.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

from jcontract.impls.claude_vision_captioner import ClaudeVisionCaptioner
from jcontract.impls.claude_vision_parser import ClaudeVisionParser
from jcontract.impls.domain_profile_registry import load_profile

SYNTHETIC_PDF = Path("eval/fixtures/synthetic_contract_tqa.pdf")


def _mock_anthropic(text: str) -> MagicMock:
    block = types.SimpleNamespace(type="text", text=text)
    usage = types.SimpleNamespace(input_tokens=1, output_tokens=1)
    client = MagicMock()
    client.messages.create.return_value = types.SimpleNamespace(content=[block], usage=usage)
    return client


def _sent_prompt(client: MagicMock) -> str:
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    return str(content[1]["text"])


def test_parser_default_vs_document_isolate_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    # Default (contract) parser → legacy un-suffixed cache filename.
    ClaudeVisionParser(
        cache_dir=cache, client=_mock_anthropic("DEMO TEXT"), max_pages=1, auto_classify=False
    ).parse(SYNTHETIC_PDF)
    # document profile parser → its own .document namespace, fresh OCR.
    doc = load_profile("document")
    doc_client = _mock_anthropic("DOC TEXT")
    pages = ClaudeVisionParser(
        cache_dir=cache,
        client=doc_client,
        max_pages=1,
        auto_classify=False,
        profile=doc,
    ).parse(SYNTHETIC_PDF)

    names = sorted(p.name for p in cache.glob("*.txt"))
    assert len(names) == 2, names  # no collision
    assert any(n.endswith(".text.txt") for n in names)  # contract legacy
    assert any(n.endswith(".text.document.txt") for n in names)  # document namespace
    # The document parser actually sent the neutral profile prompt + re-OCR'd.
    assert _sent_prompt(doc_client) == doc.ocr_text_prompt
    assert pages[0].text == "DOC TEXT"


def test_captioner_uses_profile_caption_prompt(tmp_path: Path) -> None:
    doc = load_profile("document")
    client = _mock_anthropic('{"caption_zh": "图说", "entities": []}')
    ClaudeVisionCaptioner(cache_dir=tmp_path / "cap", client=client, profile=doc).caption(
        b"image-bytes", "nearby"
    )
    sent = _sent_prompt(client)
    # Neutral caption wording, not the construction "engineering drawing".
    assert "figure, chart, or diagram" in sent
    assert "engineering drawing from a construction" not in sent
