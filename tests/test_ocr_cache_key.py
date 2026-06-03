"""Unit tests for the shared OCR-cache-key suffix helper (Enhancement E10)."""

from __future__ import annotations

from jcontract.impls._ocr_cache_key import model_cache_suffix


def test_default_model_yields_empty_suffix():
    # Equal to the parser's default → no suffix → legacy un-suffixed
    # filename → existing on-disk caches keep hitting.
    assert model_cache_suffix("haiku", "haiku") == ""
    assert model_cache_suffix("claude-sonnet-4-5", "claude-sonnet-4-5") == ""


def test_non_default_model_yields_dotted_slug():
    assert model_cache_suffix("sonnet", "haiku") == ".sonnet"
    assert model_cache_suffix("claude-sonnet-4-5", "haiku") == ".claude-sonnet-4-5"


def test_slug_is_filename_safe():
    # Slashes / colons / spaces collapse to single dashes so the suffix
    # is always a safe filename component.
    assert model_cache_suffix("vendor/model:v2 beta", "haiku") == ".vendor-model-v2-beta"


def test_leading_trailing_separators_stripped():
    assert model_cache_suffix("@@weird@@", "haiku") == ".weird"
