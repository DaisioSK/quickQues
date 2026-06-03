"""Smoke test for Phase 0 S0.2 scaffold.

Verifies that the package imports and config loads with defaults.
Any failure here means three-piece gate is misconfigured.
"""

from __future__ import annotations

import jcontract
from jcontract.config import load_app_config


def test_package_imports() -> None:
    assert jcontract.__version__ == "0.1.0"


def test_app_config_loads_with_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Clear any env that might shadow defaults from the dev shell.
    monkeypatch.delenv("APP_PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    cfg = load_app_config()
    assert cfg.app_port == 8000
    assert cfg.log_level == "INFO"
    assert cfg.qdrant_url == "http://localhost:6333"


def test_required_secret_raises_clearly(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from jcontract.config import get_anthropic_api_key

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        get_anthropic_api_key()
    except RuntimeError as e:
        # Must mention the key name; must NOT echo any value.
        assert "ANTHROPIC_API_KEY" in str(e)
        assert ".env.example" in str(e)
    else:
        raise AssertionError("Expected RuntimeError for missing required env")


def test_phase2_protocols_and_dataclasses_exported() -> None:
    """Sub-sprint p2-ss-prep finalized DrawingCaption + OCRBlock as
    dataclasses (was Protocol placeholders) and added Chunk.caption.

    Three things assertable from public surface:
      1. DrawingCaption + OCRBlock are instantiable (dataclass, not Protocol)
      2. Chunk.caption defaults to None for chunks ingested before Phase 2
      3. Re-exports through jcontract.interfaces resolve (downstream
         business code uses the single-import style)
    """
    from jcontract.interfaces import Chunk, DrawingCaption, OCRBlock

    # DrawingCaption: instantiate with both fields, then default entities.
    dc = DrawingCaption(caption_zh="这是一张桥梁防水构造图", entities=["T/PRJ/CWD/WS/2101A"])
    assert dc.caption_zh.startswith("这是")
    assert dc.entities == ["T/PRJ/CWD/WS/2101A"]
    dc_empty = DrawingCaption(caption_zh="")
    assert dc_empty.entities == []  # default_factory list, not shared mutable

    # OCRBlock: bbox is the standard (x0, y0, x1, y1) shape.
    ob = OCRBlock(text="Drawing No. T/DEMO", page_num=1, bbox=(10, 20, 200, 40), confidence=0.92)
    assert ob.page_num == 1
    assert ob.bbox == (10, 20, 200, 40)

    # Chunk.caption defaults to None (captioner-never-ran sentinel,
    # distinct from "" = ran-but-empty per DECISION-2.prep.3).
    c = Chunk(id="x:1:0", text="dummy", file="f.pdf", page=1, chunk_type="drawing")
    assert c.caption is None
