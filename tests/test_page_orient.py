"""Unit tests for the ssRT orientation probe (impls/_page_orient.py).

Strategy: ``probe_rotation`` only ever sees an injected ``ocr_fn`` — no
RapidOCR engine, no model download. The fake OCR is keyed by the EXACT
frame bytes (base + the three ``rotate_jpeg`` variants), so the tests also
prove the probe feeds ``rotate_jpeg`` output — not some other transform —
to the OCR callable. Real-engine behaviour on the frozen rotation
specimens is the ssRT e2e anchor, not a unit test.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import pytest
from PIL import Image

from jcontract.impls._page_orient import (
    GATE_MIN_SCORE,
    ROTATIONS,
    ocr_mass,
    probe_rotation,
    rotate_jpeg,
)

# A tiny non-square frame: rotation by 90/270 must swap its dimensions,
# which makes "the probe really rotated the bytes" observable. The pixel
# content must be ASYMMETRIC — a uniform image's four rotations encode to
# colliding bytes (90==270, 180==0) and would collapse the fake-OCR table.
_W, _H = 12, 30


def _base_jpeg() -> bytes:
    pixels = bytes((x * 19 + y * 7) % 256 for y in range(_H) for x in range(_W))
    buf = io.BytesIO()
    Image.frombytes("L", (_W, _H), pixels).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _dims(jpeg_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        width, height = img.size
    return width, height


class _FakeOcr:
    """ocr_fn fake: exact-bytes → (text, scores) table, with a call log."""

    def __init__(self, table: dict[bytes, tuple[str, Sequence[float]] | None]) -> None:
        self._table = table
        self.calls: list[bytes] = []

    def __call__(self, jpeg_bytes: bytes) -> tuple[str, Sequence[float]] | None:
        self.calls.append(jpeg_bytes)
        return self._table[jpeg_bytes]


def _frames(base: bytes) -> dict[int, bytes]:
    return {rot: rotate_jpeg(base, rot) for rot in ROTATIONS}


# ---------------------------------------------------------------------------
# rotate_jpeg
# ---------------------------------------------------------------------------


def test_rotate_zero_is_byte_identity():
    base = _base_jpeg()
    assert rotate_jpeg(base, 0) is base  # no decode/re-encode round trip


def test_rotate_90_270_swap_dimensions_180_preserves():
    base = _base_jpeg()
    assert _dims(rotate_jpeg(base, 90)) == (_H, _W)
    assert _dims(rotate_jpeg(base, 270)) == (_H, _W)
    assert _dims(rotate_jpeg(base, 180)) == (_W, _H)


def test_rotate_is_deterministic():
    """Same frame + same rotation = same bytes — the downstream OCR cache
    key (sha256 of the rotated frame) must be stable across runs."""
    base = _base_jpeg()
    assert rotate_jpeg(base, 90) == rotate_jpeg(base, 90)


def test_rotate_rejects_non_quarter_turns():
    with pytest.raises(ValueError):
        rotate_jpeg(_base_jpeg(), 45)


# ---------------------------------------------------------------------------
# ocr_mass
# ---------------------------------------------------------------------------


def test_ocr_mass_is_nonspace_chars_times_mean_score():
    # "ab cd" → 4 non-space chars; mean(0.8, 0.6) = 0.7 → 2.8.
    assert ocr_mass("ab cd", [0.8, 0.6]) == pytest.approx(2.8)


def test_ocr_mass_zero_for_empty_evidence():
    assert ocr_mass("", []) == 0.0
    assert ocr_mass("text but no boxes", []) == 0.0
    assert ocr_mass("", [0.9]) == 0.0


# ---------------------------------------------------------------------------
# probe_rotation — gate
# ---------------------------------------------------------------------------


def test_good_page_is_gated_out_after_one_probe():
    """min_score >= threshold → rotation 0, exactly ONE OCR call (no 4x)."""
    base = _base_jpeg()
    ocr = _FakeOcr({base: ("clean upright text", [0.99, GATE_MIN_SCORE])})

    rotation, evidence = probe_rotation(base, ocr)

    assert rotation == 0
    assert ocr.calls == [base]
    assert evidence["gated"] is False
    assert list(evidence["probes"]) == ["0"]


def test_low_min_score_triggers_four_direction_probe():
    base = _base_jpeg()
    frames = _frames(base)
    ocr = _FakeOcr(
        {
            frames[0]: ("short frags", [0.95, 0.60]),  # gated: min 0.60 < 0.756
            frames[90]: ("a much longer confidently read line of text", [0.98, 0.97]),
            frames[180]: ("junk", [0.5]),
            frames[270]: ("junk", [0.5]),
        }
    )

    rotation, evidence = probe_rotation(base, ocr)

    assert rotation == 90
    assert evidence["gated"] is True
    assert set(evidence["probes"]) == {"0", "90", "180", "270"}
    assert [frames[r] for r in ROTATIONS] == ocr.calls  # probed via rotate_jpeg


def test_zero_box_page_is_gated_through_but_stays_zero_on_silence():
    """No boxes = no orientation evidence → probe; all-silent frames keep 0."""
    base = _base_jpeg()
    frames = _frames(base)
    ocr = _FakeOcr({frame: ("", []) for frame in frames.values()})

    rotation, evidence = probe_rotation(base, ocr)

    assert rotation == 0
    assert evidence["gated"] is True
    assert len(ocr.calls) == 4


# ---------------------------------------------------------------------------
# probe_rotation — winner selection
# ---------------------------------------------------------------------------


def test_dead_heat_prefers_no_rotation():
    base = _base_jpeg()
    frames = _frames(base)
    ocr = _FakeOcr({frame: ("same mass", [0.6]) for frame in frames.values()})

    rotation, _ = probe_rotation(base, ocr)

    assert rotation == 0


def test_winner_below_improvement_margin_keeps_zero():
    """A non-zero direction barely ahead (< factor) is noise, not rotation."""
    base = _base_jpeg()
    frames = _frames(base)
    ocr = _FakeOcr(
        {
            frames[0]: ("aaaaaaaaaa", [0.70]),  # mass 7.0, gated
            frames[90]: ("aaaaaaaaaaa", [0.70]),  # mass 7.7 = 1.10x, NOT > margin
            frames[180]: ("a", [0.5]),
            frames[270]: ("a", [0.5]),
        }
    )

    rotation, _ = probe_rotation(base, ocr, improvement_factor=1.10)

    assert rotation == 0


def test_blank_at_zero_rotates_when_any_direction_reads_text():
    """mass(0)=0: any direction with real text clears the margin (x*0=0)."""
    base = _base_jpeg()
    frames = _frames(base)
    ocr = _FakeOcr(
        {
            frames[0]: ("", []),
            frames[90]: ("", []),
            frames[180]: ("recovered upside down text", [0.9]),
            frames[270]: ("", []),
        }
    )

    rotation, _ = probe_rotation(base, ocr)

    assert rotation == 180


# ---------------------------------------------------------------------------
# probe_rotation — engine failure (None from ocr_fn)
# ---------------------------------------------------------------------------


def test_baseline_probe_failure_degrades_to_zero_with_error_flag():
    base = _base_jpeg()
    ocr = _FakeOcr({base: None})

    rotation, evidence = probe_rotation(base, ocr)

    assert rotation == 0
    assert evidence["engine_error"] is True
    assert len(ocr.calls) == 1


def test_mid_probe_failure_degrades_to_zero_with_error_flag():
    """A missing direction could have been the winner — never decide on a
    partial comparison (the caller must not cache this run)."""
    base = _base_jpeg()
    frames = _frames(base)
    ocr = _FakeOcr(
        {
            frames[0]: ("low quality", [0.5]),
            frames[90]: None,
            frames[180]: ("would have won easily with this much text", [0.99]),
            frames[270]: ("x", [0.5]),
        }
    )

    rotation, evidence = probe_rotation(base, ocr)

    assert rotation == 0
    assert evidence["engine_error"] is True
