"""Unit tests for the ssCfg pagefix policy (PagefixPolicy + load_policy).

Strategy:
- Regression guard: the built-in default policy is byte-equal to the live
  module constants — this is the zero-behaviour-change contract, pinned so a
  drifted constant fails loudly.
- Loader behaviour: the shipped framework default档 round-trips to that same
  default; a partial YAML overrides only what it names; bad input raises;
  JCONTRACT_PAGEFIX_POLICY / explicit-path resolution behaves.
- Flip proof: a policy that tightens a classify threshold flips a page's
  classify_page_v2 verdict (default → drawing on the same pixels + boxes),
  while the no-policy / default-policy path keeps the verbatim verdict.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from jcontract.impls._page_classify import (
    V2_FILLED_DARK_RATIO,
    V2_FRAGMENT_BOX_FRAC,
    V2_SPARSE_COVER,
    V2_SPARSE_DARK,
    classify_page_v2,
)
from jcontract.impls._page_orient import GATE_MIN_SCORE, IMPROVEMENT_FACTOR
from jcontract.impls._pagefix_policy import (
    DEFAULT_POLICY_NAME,
    PagefixPolicy,
    load_policy,
)
from jcontract.impls.rapidocr_parser import RESCUE_DPI, RESCUE_MIN_SCORE

# The framework default档 ships at the repo-root profiles/ dir.
SHIPPED_POLICY = Path(__file__).resolve().parents[1] / "profiles" / "pagefix-policy.yaml"


def _jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _page_with_ink(ink_fraction: float, size: tuple[int, int] = (800, 1000)) -> bytes:
    img = Image.new("L", size, 255)
    band_height = int(size[1] * ink_fraction)
    if band_height:
        ImageDraw.Draw(img).rectangle((0, 0, size[0], band_height), fill=0)
    return _jpeg(img)


# ---------------------------------------------------------------------------
# Zero-behaviour-change contract: default policy == live module constants
# ---------------------------------------------------------------------------


def test_default_policy_thresholds_are_byte_equal_to_constants():
    """The no-arg PagefixPolicy must equal the current hard-coded constants.

    This is the ssCfg hard constraint (DECISION-pm.10): the default policy is
    byte-for-byte the present behaviour. Imported constants, not transcribed
    literals, so a constant that drifts breaks here — exactly where it should.
    """
    p = PagefixPolicy()
    assert p.gate_min_score == GATE_MIN_SCORE
    assert p.improvement_factor == IMPROVEMENT_FACTOR
    assert p.v2_sparse_cover == V2_SPARSE_COVER
    assert p.v2_sparse_dark == V2_SPARSE_DARK
    assert p.v2_fragment_box_frac == V2_FRAGMENT_BOX_FRAC
    assert p.v2_filled_dark_ratio == V2_FILLED_DARK_RATIO
    assert p.rescue_dpi == RESCUE_DPI
    assert p.rescue_min_score == RESCUE_MIN_SCORE


def test_default_policy_valve_toggles_are_decoupled_defaults():
    """DECISION-pm.3: rotate/rescue ON, regions/v2 OFF."""
    p = PagefixPolicy()
    assert p.rotate is True
    assert p.rescue is True
    assert p.regions is False
    assert p.needs_vision_v2 is False


def test_shipped_default_yaml_round_trips_to_the_default_policy():
    """profiles/pagefix-policy.yaml must decode to the in-code default exactly."""
    from_file = load_policy(str(SHIPPED_POLICY))
    assert from_file == PagefixPolicy()


def test_load_policy_missing_default_name_returns_in_code_default(monkeypatch, tmp_path):
    """The built-in name never raises even with no file — the dataclass is the
    authoritative fallback (byte-equal to current behaviour)."""
    monkeypatch.setenv("JCONTRACT_PAGEFIX_POLICY", str(tmp_path))  # empty dir
    load_policy.cache_clear()
    assert load_policy(DEFAULT_POLICY_NAME) == PagefixPolicy()


# ---------------------------------------------------------------------------
# Loader: partial override, env/path resolution, malformed input
# ---------------------------------------------------------------------------


def test_partial_yaml_overrides_only_named_keys(tmp_path):
    yaml_path = tmp_path / "tight.yaml"
    yaml_path.write_text(
        "valves:\n  needs_vision_v2: true\nssvr:\n  v2_fragment_box_frac: 0.0005\n",
        encoding="utf-8",
    )
    p = load_policy(str(yaml_path))
    # Named keys changed…
    assert p.needs_vision_v2 is True
    assert p.v2_fragment_box_frac == 0.0005
    # …everything else keeps the byte-equal default.
    assert p.v2_sparse_cover == V2_SPARSE_COVER
    assert p.rotate is True
    assert p.rescue_dpi == RESCUE_DPI


def test_env_dir_resolves_bare_name(monkeypatch, tmp_path):
    (tmp_path / "mypolicy.yaml").write_text("ssrt:\n  gate_min_score: 0.9\n", encoding="utf-8")
    monkeypatch.setenv("JCONTRACT_PAGEFIX_POLICY", str(tmp_path))
    load_policy.cache_clear()
    assert load_policy("mypolicy").gate_min_score == 0.9


def test_unknown_named_policy_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("JCONTRACT_PAGEFIX_POLICY", str(tmp_path))
    load_policy.cache_clear()
    with pytest.raises(ValueError, match="Unknown pagefix policy"):
        load_policy("does-not-exist")


def test_non_mapping_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_policy(str(bad))


def test_non_bool_toggle_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("valves:\n  rotate: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a bool"):
        load_policy(str(bad))


# ---------------------------------------------------------------------------
# Flip proof: a tightened policy threshold flips a classify verdict
# ---------------------------------------------------------------------------


def test_tightened_fragment_threshold_flips_page_to_drawing():
    """A full-line-box page is 'text' under default thresholds; raising
    fragment_box_frac (require larger boxes to count as text) flips it to
    'drawing' on the SAME pixels + box stats — proves config drives the
    decision while the default is unchanged.
    """
    inked = _page_with_ink(0.08)
    boxes, coverage = 50, 0.35  # mean box area .007 — comfortably "text"

    # Default thresholds → text.
    default = classify_page_v2(inked, boxes=boxes, box_coverage=coverage)
    assert default == "text"

    # Tightened policy: demand mean box area ≥ .01 to be text → .007 < .01 → drawing.
    tight = PagefixPolicy(v2_fragment_box_frac=0.01)
    flipped = classify_page_v2(
        inked,
        boxes=boxes,
        box_coverage=coverage,
        sparse_cover=tight.v2_sparse_cover,
        sparse_dark=tight.v2_sparse_dark,
        fragment_box_frac=tight.v2_fragment_box_frac,
        filled_dark_ratio=tight.v2_filled_dark_ratio,
    )
    # default was "text", flipped is "drawing" → config flipped the verdict.
    assert flipped == "drawing"


def test_default_policy_classify_matches_no_arg_call():
    """Injecting the DEFAULT policy thresholds equals the no-arg call (parity)."""
    inked = _page_with_ink(0.08)
    p = PagefixPolicy()
    with_policy = classify_page_v2(
        inked,
        boxes=50,
        box_coverage=0.35,
        sparse_cover=p.v2_sparse_cover,
        sparse_dark=p.v2_sparse_dark,
        fragment_box_frac=p.v2_fragment_box_frac,
        filled_dark_ratio=p.v2_filled_dark_ratio,
    )
    without = classify_page_v2(inked, boxes=50, box_coverage=0.35)
    assert with_policy == without
