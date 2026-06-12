"""Page-orientation probe — pick the 90°-multiple rotation that OCRs best (ssRT).

What:
    ``probe_rotation`` takes one rendered page (JPEG bytes) plus an OCR
    callable and decides which of the four 90°-multiple rotations
    (0/90/180/270, degrees **counter-clockwise**) yields the most readable
    frame. Quality is "OCR mass" = non-whitespace character count × mean
    per-box recognition score. ``rotate_jpeg`` is the shared frame
    transform every downstream consumer (parser OCR, caption lane,
    table-preview) uses, so "rotation=90" means the same pixels everywhere.

Why:
    The v6 gallery review found that the worst flagged pages are mostly
    CLEAN scans stored sideways/upside-down — the OCR engine reads them as
    ordered fragments ("OCR 同水平线放一行" on vertical text). Re-feeding
    the engine an upright frame recovers them at zero model cost. Detection
    is by experiment, not metadata: scanned PDFs carry no trustworthy
    /Rotate flag, so we let the OCR engine itself vote — the upright frame
    measurably produces more confident characters (live-verified on the two
    frozen rotation specimens, see DECISION-pl.10).

Context:
    Mechanism for the ssRT opt-in lane (``--auto-rotate``); nothing here
    runs by default. Only 90° multiples — arbitrary-angle deskew is out of
    scope (FORESHADOW-pl.1). Consumers downstream of the probe:
    RapidOcrParser (OCR on the upright frame + rotation sidecar cache),
    IngestPipeline._attach_captions (VLM sees the upright frame),
    table-preview (structure model sees the upright frame).

Shared gate semantics (ssGE/ssVR/ssHD alignment):
    "Page quality is low" here means ``min_score < GATE_MIN_SCORE`` — the
    same per-box min-recognition-score signal (and the same 0.756 value)
    the frozen W6 quality protocol uses for flagging, so the probe lane and
    the quality-report lane agree on WHICH pages are suspect. Sibling
    sub-sprints reuse the signal semantics but own their thresholds.
    [DECISION-pl.11]
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# Rotations are degrees COUNTER-CLOCKWISE (PIL Transpose semantics), probed
# in this order; ties resolve to the earliest entry, i.e. prefer no-op.
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)

# Trigger gate: probe the three extra rotations only when the rotation-0
# frame's min per-box score is below this. 0.756 is the frozen W6 flagging
# threshold (min_score:0.756) — measured 2026-06-12: both frozen rotation
# specimens land under it at rotation 0 (TQA p.9 = 0.6132, 3of9 p.200 =
# 0.6262) while mean_score stays ~0.95-0.96 on the SAME sideways pages
# (the engine's cls stage confidently reads fragments), so mean_score
# cannot gate. [DECISION-pl.11]
GATE_MIN_SCORE = 0.756

# A non-zero rotation wins only when its OCR mass beats the rotation-0 mass
# by this factor. Measured margins on the frozen specimens: upright/0 =
# 1.49 (TQA p.9) and 1.22 (3of9 p.200); upright controls never produced a
# non-zero winner at all — 1.10 keeps both specimens converting while
# protecting genuinely-upright low-quality pages from noise flips.
# [DECISION-pl.10]
IMPROVEMENT_FACTOR = 1.10

# ocr_fn contract: JPEG bytes in → (assembled text, per-box recognition
# scores) out, or None when the OCR engine itself failed (transient — the
# caller must NOT cache a decision built on a failed probe).
OcrFn = Callable[[bytes], "tuple[str, Sequence[float]] | None"]

_TRANSPOSE: dict[int, Image.Transpose] = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def rotate_jpeg(jpeg_bytes: bytes, rotation: int, *, jpeg_quality: int = 85) -> bytes:
    """Rotate a JPEG frame by a 90° multiple (counter-clockwise), re-encoded.

    rotation=0 returns the input bytes UNCHANGED (no decode/re-encode round
    trip) — the identity case must preserve the original cache key. The
    re-encode for non-zero rotations uses the same default quality as the
    render path (85), and PIL encodes identical pixels deterministically,
    so a given (page, rotation) always produces the same bytes → the same
    downstream OCR cache key. The rotated frame's sha256 is therefore a NEW
    content-addressed key, naturally isolated from the original frame's
    cache entries — zero namespace collision by construction.
    """
    if rotation == 0:
        return jpeg_bytes
    if rotation not in _TRANSPOSE:
        raise ValueError(f"rotation must be one of {ROTATIONS}, got {rotation}")
    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        rotated = img.transpose(_TRANSPOSE[rotation])
    buf = io.BytesIO()
    rotated.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def ocr_mass(text: str, scores: Sequence[float]) -> float:
    """OCR mass of one probe: non-whitespace chars × mean per-box score.

    What: the quality scalar the four-direction comparison ranks by.
    Why this combination [DECISION-pl.10]: char count alone rewards the
    engine hallucinating fragments on sideways text; mean score alone
    saturates (~0.95 even on sideways frames, measured). Their product
    tracks "how much confidently-read text this frame yields" and, on the
    frozen specimens, ranks the truly upright frame first — agreeing with
    the difflib-vs-gold ranking (0.2101 vs ≤0.0587 on TQA p.9; 0.4085 vs
    ≤0.1721 on 3of9 p.200). A zero-box or empty-text probe has zero mass.
    """
    if not scores:
        return 0.0
    chars = sum(1 for ch in text if not ch.isspace())
    return chars * (sum(float(s) for s in scores) / len(scores))


def _probe_record(text: str, scores: Sequence[float]) -> dict[str, Any]:
    """One rotation's evidence row: box/char counts, score stats, mass."""
    score_list = [float(s) for s in scores]
    return {
        "boxes": len(score_list),
        "chars": sum(1 for ch in text if not ch.isspace()),
        "mean_score": round(sum(score_list) / len(score_list), 4) if score_list else None,
        "min_score": round(min(score_list), 4) if score_list else None,
        "mass": round(ocr_mass(text, score_list), 1),
    }


def probe_rotation(
    jpeg_bytes: bytes,
    ocr_fn: OcrFn,
    *,
    gate_min_score: float = GATE_MIN_SCORE,
    improvement_factor: float = IMPROVEMENT_FACTOR,
    jpeg_quality: int = 85,
) -> tuple[int, dict[str, Any]]:
    """Decide the best 90°-multiple rotation for one rendered page.

    Flow:
      1. OCR the frame as-is (rotation 0). This probe is never wasted —
         a cache-aware ``ocr_fn`` persists it as the page's normal OCR.
      2. Gate: when the rotation-0 frame is NOT low quality
         (``min_score >= gate_min_score``), stop — rotation 0, one OCR
         pass, zero extra cost. Probing every page would 4x the corpus
         OCR bill for the ~50% of pages that are fine. A zero-box frame
         (min_score undefined) IS gated through: no score evidence means
         no proof the frame is upright. [DECISION-pl.11]
      3. Probe 90/180/270 via ``rotate_jpeg`` and rank all four by
         ``ocr_mass``; ties prefer the earlier rotation (0 first).
      4. A non-zero winner must beat rotation-0 mass by
         ``improvement_factor``, else keep 0. [DECISION-pl.10]

    Returns ``(rotation, evidence)``. ``evidence["engine_error"] = True``
    marks a run where any probe's OCR failed — the decision degrades to 0
    and callers must not cache it (transient failures retry next run).
    """
    evidence: dict[str, Any] = {
        "gate_signal": "min_score",
        "gate_threshold": gate_min_score,
        "gated": False,
        "probes": {},
    }

    # Step 1 — baseline probe on the frame as rendered.
    base = ocr_fn(jpeg_bytes)
    if base is None:
        evidence["engine_error"] = True
        return 0, evidence
    text0, scores0 = base
    evidence["probes"]["0"] = _probe_record(text0, scores0)
    masses: dict[int, float] = {0: ocr_mass(text0, scores0)}

    # Step 2 — trigger gate: only low-quality frames pay the 3 extra probes.
    min0 = min(float(s) for s in scores0) if scores0 else None
    if min0 is not None and min0 >= gate_min_score:
        return 0, evidence
    evidence["gated"] = True

    # Step 3 — probe the three rotated frames.
    for rotation in ROTATIONS[1:]:
        probed = ocr_fn(rotate_jpeg(jpeg_bytes, rotation, jpeg_quality=jpeg_quality))
        if probed is None:
            # One failed probe poisons the whole comparison: a missing
            # direction could have been the winner. Degrade to "no
            # rotation" and signal the caller not to cache.
            evidence["engine_error"] = True
            return 0, evidence
        text_r, scores_r = probed
        evidence["probes"][str(rotation)] = _probe_record(text_r, scores_r)
        masses[rotation] = ocr_mass(text_r, scores_r)

    # Step 4 — rank; strict > keeps the earliest rotation on ties (0 wins
    # a dead heat), then apply the anti-noise improvement margin.
    best = 0
    for rotation in ROTATIONS:
        if masses[rotation] > masses[best]:
            best = rotation
    if best != 0 and masses[best] <= masses[0] * improvement_factor:
        logger.info(
            "page_orient.improvement_below_margin",
            best_rotation=best,
            best_mass=round(masses[best], 1),
            base_mass=round(masses[0], 1),
            factor=improvement_factor,
        )
        best = 0
    return best, evidence
