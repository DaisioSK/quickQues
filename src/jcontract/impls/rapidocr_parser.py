"""RapidOcrParser — PDFParser via RapidOCR (PP-OCRv5 ONNX, local CPU).

Why this exists:
- Enhancement E3 (sub-sprint ssLC): every existing vision parser sends page
  images to a third party (Anthropic / DeepSeek) or burns subscription
  quota. This vendor OCRs entirely on the local CPU — zero cost, zero data
  egress — completing the "fully local stack" story alongside the local
  answerer (ssLA) and local captioner (ssLB).
- Engine choice: ``rapidocr`` 3.x (the actively maintained successor of the
  legacy ``rapidocr-onnxruntime`` package, frozen at 1.4.4 since 2025-01)
  running PaddleOCR PP-OCRv5 mobile models through CPU onnxruntime.
  PaddleOCR itself was rejected: paddlepaddle wheel bulk + unverified
  sm_120 GPU story. [DECISION-ls.30 dev-sprint v4 §13]

Architecture:
- Identical render path to the other vision parsers: pypdfium2 → PIL → JPEG
  @ 150 DPI q=85, all pdfium calls behind the process-global lock
  (``_pdfium_render``, DECISION-ab3.46). Same JPEG bytes → same sha256 →
  cross-vendor cache-key compatibility.
- Cache layout: ``data/ocr_cache/rapidocr-<sha256>.text[.<model>].txt`` —
  the ``rapidocr-`` filename prefix isolates this vendor's namespace from
  the Claude (``<hash>.text*.txt``) and DeepSeek (``deepseek-v4-<hash>``)
  entries living in the same dir (mirrors DECISION-1.10.3). The model slug
  suffix (suppressed for the default ``ppocrv5-mobile``) namespaces a
  future server-model run, via the shared ``model_cache_suffix`` helper.
- No DomainProfile parameter: RapidOCR takes no prompt — output depends
  only on pixels + ONNX weights, so a profile cannot change it and must
  not fork the cache namespace (unlike the LLM vendors, where the prompt
  is profile-driven).

Offline property (verified 2026-06-11): PP-OCRv4 mobile models ship inside
the wheel; selecting PP-OCRv5 triggers a one-time ~20MB download
(det 4.6MB + rec 15.9MB from modelscope.cn) into
``site-packages/rapidocr/models/``. After that, init + OCR succeed with all
network access blocked (tested behind a dead proxy).

Reading order: RapidOCR returns (box, text, score) triples in detection
order. We re-assemble by banding boxes into visual lines (y-top within half
a box height) and sorting left-to-right within each line. Good enough for
the linear TQA/letter pages this project ingests; complex multi-column /
table layouts are out of scope (FORESHADOW-ls.3, dev-sprint v4 §13).

Cost / speed: $0, ~1s/page CPU (vs 10-15s/page + quota on claude-cli-vision).
Fidelity is lower than LLM vision OCR — measured separately in the L5
OCR-fidelity comparison (docs/localstack-ocr-compare.md, project repo).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pypdfium2 as pdfium
import structlog

from jcontract.impls._ocr_cache_key import model_cache_suffix
from jcontract.impls._page_classify import _classify_page
from jcontract.impls._page_orient import probe_rotation, rotate_jpeg
from jcontract.impls._pdfium_render import render_page_jpeg
from jcontract.interfaces import PageKind, ParsedPage

logger = structlog.get_logger(__name__)

# Suppress noisy pypdfium2 INFO logs during rendering (same as other parsers).
logging.getLogger("pypdfium2").setLevel(logging.WARNING)

# Same render geometry as every other vision parser — identical JPEG bytes
# are what make the sha256 cache key comparable across vendors.
DEFAULT_DPI = 150
DEFAULT_JPEG_QUALITY = 85

# PP-OCRv5 "mobile" det/rec pair: ~20MB total, ~1s/page on CPU. The "server"
# pair is ~5x heavier for marginal accuracy on clean scans — selectable via
# the constructor; a non-default choice gets its own cache namespace.
DEFAULT_MODEL_TYPE = "mobile"
_MODEL_SLUG_TEMPLATE = "ppocrv5-{model_type}"

# ssQA: per-box recognition scores below this count as "low confidence" in
# the metrics sidecar (`low_score_ratio`). 0.7 is part of the FROZEN
# pre-registered signal list (dev-sprint v5 §预注册评测协议 3) — changing it
# would invalidate the L5 calibration, so it is a module constant, not a knob.
LOW_SCORE_THRESHOLD = 0.7

# ssQA garbled-text heuristic: characters we EXPECT on these scans —
# printable ASCII, CJK ideographs (incl. Ext-A), CJK/fullwidth punctuation,
# common general punctuation (–—‘’“”…·), and whitespace. Anything else
# (e.g. ª, ¤, stray Greek/Cyrillic, box-drawing junk) is what low-quality
# OCR of noisy scans typically emits → counts toward `garbled_ratio`.
_EXPECTED_CHAR_RANGES: tuple[tuple[int, int], ...] = (
    (0x20, 0x7E),  # printable ASCII
    (0x2010, 0x2027),  # general punctuation: dashes, quotes, ellipsis
    (0x3000, 0x303F),  # CJK symbols & punctuation
    (0x3400, 0x4DBF),  # CJK Ext-A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xFF00, 0xFFEF),  # fullwidth forms
)


def _is_expected_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _EXPECTED_CHAR_RANGES)


def _page_metrics(page_num: int, scores: Sequence[float], text: str) -> dict[str, Any]:
    """Compute the per-page OCR quality metrics record (sidecar schema, ssQA).

    Schema (frozen with the pre-registered signal list, DECISION-cq.21):
      page_num         page the bytes were first seen on (informational —
                       the sidecar itself is content-addressed)
      boxes            number of recognised text boxes
      scores           raw per-box recognition scores (full list — the
                       sidecar is the lossless record; reports aggregate)
      mean_score       mean of scores            (null when boxes == 0)
      min_score        min of scores             (null when boxes == 0)
      low_score_ratio  share of boxes < 0.7      (null when boxes == 0)
      alnum_ratio      alphanumeric share of non-whitespace chars
                       (str.isalnum — covers CJK)   (null when no chars)
      garbled_ratio    share of non-whitespace chars outside the expected
                       charset (see _EXPECTED_CHAR_RANGES) (null when no chars)

    Why null (not 0.0) for undefined: a zero-box page has NO score evidence —
    forcing 0.0 would make it look maximally bad on mean_score and perfectly
    good on low_score_ratio at the same time. Callers treat null as "signal
    unavailable"; empty pages are still catchable via the `boxes` signal.
    """
    score_list = [float(s) for s in scores]
    n = len(score_list)
    chars = [ch for ch in text if not ch.isspace()]
    n_chars = len(chars)
    return {
        "page_num": page_num,
        "boxes": n,
        "scores": score_list,
        "mean_score": (sum(score_list) / n) if n else None,
        "min_score": min(score_list) if n else None,
        "low_score_ratio": (sum(1 for s in score_list if s < LOW_SCORE_THRESHOLD) / n)
        if n
        else None,
        "alnum_ratio": (sum(1 for ch in chars if ch.isalnum()) / n_chars) if n_chars else None,
        "garbled_ratio": (sum(1 for ch in chars if not _is_expected_char(ch)) / n_chars)
        if n_chars
        else None,
    }


def _assemble_reading_order(boxes: Sequence[Sequence[Sequence[float]]], txts: Sequence[str]) -> str:
    """Re-assemble RapidOCR (box, text) pairs into reading-order plain text.

    What: bands boxes into visual lines (a box joins the current line when
    its top edge is within half its own height of the line anchor), sorts
    each line left-to-right, joins blocks with a space and lines with a
    newline.

    Why this simple rule: project pages are linear (letters, Q&A sheets);
    per the ssLC scope line, complex layouts (multi-column, tables) are
    explicitly deferred — FORESHADOW-ls.3. Boxes are 4-point quads
    ``[[x,y] * 4]`` in pixel space; we reduce each to (y_top, x_left,
    height) and never need the full geometry.
    """
    blocks: list[tuple[float, float, float, str]] = []
    for box, txt in zip(boxes, txts, strict=True):
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        blocks.append((min(ys), min(xs), max(ys) - min(ys), str(txt)))

    # Primary sort by top edge gives us a vertical sweep; line banding below
    # fixes the few-pixel jitter between boxes that share a visual row.
    blocks.sort(key=lambda b: (b[0], b[1]))

    lines: list[list[tuple[float, str]]] = []
    line_anchor_y: float | None = None
    for y_top, x_left, height, txt in blocks:
        # New line when this box starts clearly below the current anchor.
        # max(height, 1.0) guards degenerate zero-height boxes.
        if line_anchor_y is None or (y_top - line_anchor_y) > max(height, 1.0) * 0.5:
            lines.append([])
            line_anchor_y = y_top
        lines[-1].append((x_left, txt))

    return "\n".join(" ".join(txt for _, txt in sorted(line)) for line in lines)


class RapidOcrParser:
    """PDFParser that OCRs scanned PDFs locally via RapidOCR (PP-OCRv5, CPU).

    Zero API key, zero network after the one-time model download, $0
    marginal cost. Drop-in alternative to the cloud vision parsers (same
    Protocol, same cache dir). Choose via CLI ``--parser rapidocr``.
    """

    backend: ClassVar[str] = "rapidocr"
    # Cache filename prefix isolates this vendor from Claude / DeepSeek
    # entries in the shared ocr_cache/ dir (mirrors DECISION-1.10.3).
    cache_prefix: ClassVar[str] = "rapidocr"

    def __init__(
        self,
        *,
        cache_dir: Path = Path("data/ocr_cache"),
        dpi: int = DEFAULT_DPI,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        model_type: str = DEFAULT_MODEL_TYPE,
        max_pages: int | None = None,
        engine: Callable[[bytes], Any] | None = None,
        auto_classify: bool = True,
        auto_rotate: bool = False,
    ) -> None:
        # Tests inject a fake ``engine`` callable; production lazily builds
        # the real RapidOCR pipeline on first use (_ensure_engine) so that
        # importing this module — or constructing the parser — never pays
        # the opencv/onnxruntime import or the model-download cost.
        self._engine = engine
        self._cache_dir = cache_dir
        self._dpi = dpi
        self._jpeg_quality = jpeg_quality
        self._model_type = model_type
        self._max_pages = max_pages
        # ssCL: page-kind classification (shared text-vs-drawing heuristic)
        # so drawing pages enter the --caption lane. Unlike the LLM vendors
        # the verdict does NOT change what this engine OCRs (no prompt) and
        # does NOT touch the cache key — it only sets ParsedPage.page_kind.
        # auto_classify=False forces "text" for every page (eval baselines).
        self._auto_classify = auto_classify
        # ssRT: opt-in orientation probe. When True, pages whose rotation-0
        # frame is low quality (min_score < GATE_MIN_SCORE, DECISION-pl.11)
        # get the four-direction OCR-mass comparison; the winning rotation
        # is cached in a sidecar keyed by the ORIGINAL frame's sha256 and
        # the upright frame feeds the normal OCR path (its own sha256 = its
        # own cache key — zero collision with the original namespace).
        # Default False = zero behaviour change for every existing caller.
        self._auto_rotate = auto_rotate
        # RapidOCR's pipeline mutates per-call internal buffers; nothing in
        # the codebase calls this vendor concurrently today (batch-ingest
        # only wires the network vendors), but the lock makes _ocr_jpeg
        # thread-safe by construction — same belt-and-braces stance as the
        # pdfium global lock.
        self._engine_call_lock = threading.Lock()
        # Default model keeps the bare "rapidocr-<hash>.text.txt" filename;
        # a non-default model (e.g. ppocrv5-server) gets ".ppocrv5-server"
        # appended — shared suffix logic with the Claude vendors (E10).
        self._model_suffix = model_cache_suffix(
            _MODEL_SLUG_TEMPLATE.format(model_type=model_type),
            _MODEL_SLUG_TEMPLATE.format(model_type=DEFAULT_MODEL_TYPE),
        )

    def _ensure_engine(self) -> Callable[[bytes], Any]:
        """Lazily build the RapidOCR pipeline (first call only).

        Why lazy: ``import rapidocr`` drags in opencv (~100MB) and the
        first construction may download PP-OCRv5 ONNX models (~20MB).
        Neither belongs in CLI cold-start when the user picked another
        parser — same pattern as DeepSeekV4Parser._ensure_client.
        """
        if self._engine is None:
            from rapidocr import OCRVersion, RapidOCR

            # Explicit PP-OCRv5 for both det and rec: the package default is
            # still the bundled PP-OCRv4 models; v5 is the E3 target (better
            # printed-English spacing on our scans, verified 2026-06-11).
            # The cls (orientation) stage keeps its bundled default — v5
            # ships no cls model.
            self._engine = RapidOCR(
                params={
                    "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                }
            )
        return self._engine

    def parse(self, pdf_path: Path) -> list[ParsedPage]:
        """Render + OCR every page (up to ``max_pages``) and return ParsedPage list."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            page_count = min(total_pages, self._max_pages) if self._max_pages else total_pages
            logger.info(
                "rapidocr_parser.start",
                pdf=pdf_path.name,
                total_pages=total_pages,
                will_process=page_count,
                model_type=self._model_type,
                dpi=self._dpi,
            )

            pages: list[ParsedPage] = []
            for page_idx in range(page_count):
                page_num = page_idx + 1  # 1-indexed per ParsedPage contract
                pages.append(self._parse_page(pdf[page_idx], page_num, pdf_path.name))
            return pages
        finally:
            pdf.close()

    def _parse_page(self, page: pdfium.PdfPage, page_num: int, pdf_name: str) -> ParsedPage:
        """Render one page via the shared serialized helper, classify + OCR it.

        Render goes through the process-global pdfium lock so the JPEG
        bytes — payload AND cache key — are byte-identical to what every
        other vendor produces for the same page (DECISION-ab3.46).

        ssCL: the same rendered JPEG feeds the shared text-vs-drawing
        heuristic; the verdict is recorded on ``ParsedPage.page_kind`` so
        the chunker can emit drawing chunks for the --caption lane. OCR
        text and cache layout are completely unaffected by the verdict.

        ssRT: with ``auto_rotate`` on, the orientation decision happens
        FIRST and everything downstream — OCR, classification, the
        ParsedPage — sees the upright frame; the chosen rotation rides
        along on ``ParsedPage.rotation`` so the caption lane can rotate
        its own render the same way.
        """
        jpeg_bytes = render_page_jpeg(page, dpi=self._dpi, jpeg_quality=self._jpeg_quality)
        rotation = 0
        if self._auto_rotate:
            rotation = self.resolve_rotation(jpeg_bytes, page_num, pdf_name)
            if rotation:
                jpeg_bytes = rotate_jpeg(jpeg_bytes, rotation, jpeg_quality=self._jpeg_quality)
        text = self._ocr_jpeg(jpeg_bytes, page_num, pdf_name)
        return ParsedPage(
            page_num=page_num,
            text=text,
            page_kind=self._page_kind(jpeg_bytes, page_num, pdf_name),
            rotation=rotation,
        )

    def _page_kind(self, jpeg_bytes: bytes, page_num: int, pdf_name: str) -> PageKind:
        """auto_classify-aware classification with the safe-text fallback.

        Mirrors the LLM vendors' belt-and-braces stance: a raising
        (monkey-patched) classifier must not lose the page — fall back to
        "text" and log.
        """
        if not self._auto_classify:
            return "text"
        try:
            return self._classify(jpeg_bytes)
        except Exception:  # noqa: BLE001
            logger.warning(
                "rapidocr_parser.classify_raised_fallback_text",
                pdf=pdf_name,
                page=page_num,
            )
            return "text"

    def _classify(self, jpeg_bytes: bytes) -> PageKind:
        """Indirection so tests can monkeypatch classification on the instance.

        Defers to the shared ``_page_classify._classify_page`` heuristic —
        same calibration thresholds as the Claude/DeepSeek vendors (N=2 /
        §5.3: tune once, every vendor follows).
        """
        return _classify_page(jpeg_bytes)

    def _text_cache_path(self, cache_key: str) -> Path:
        """OCR text cache file for these JPEG bytes.

        Vendor prefix + "text" kind (this vendor never produces drawing
        captions) + model suffix. Profile deliberately absent — see module
        docstring.
        """
        return self._cache_dir / f"{self.cache_prefix}-{cache_key}.text{self._model_suffix}.txt"

    def _metrics_path(self, cache_key: str) -> Path:
        """Quality-metrics sidecar for these JPEG bytes (ssQA).

        Same namespace rules as the .txt: vendor prefix + content hash +
        model suffix — `rapidocr-<sha256>.metrics[.<model>].json`. A
        non-default model OCRs differently, so its scores live in their
        own sidecar namespace exactly like its text does.
        """
        return self._cache_dir / f"{self.cache_prefix}-{cache_key}.metrics{self._model_suffix}.json"

    def _rotation_path(self, cache_key: str) -> Path:
        """Rotation-decision sidecar for these ORIGINAL frame bytes (ssRT).

        `rapidocr-<sha256>.rotation[.<model>].json` — keyed by the
        as-rendered frame's hash (the only stable identity the page has
        before the decision exists). Model-suffixed like the .txt/.metrics
        sidecars: the probe ranks ENGINE output, so a different model may
        legitimately decide differently. [DECISION-pl.12]
        """
        return (
            self._cache_dir / f"{self.cache_prefix}-{cache_key}.rotation{self._model_suffix}.json"
        )

    def resolve_rotation(self, jpeg_bytes: bytes, page_num: int, pdf_name: str) -> int:
        """Cached orientation decision for one rendered page (ssRT).

        Sidecar-first: a stored decision is returned without any OCR (the
        whole point — re-ingest must not re-pay the 4x probe). On a miss,
        ``probe_rotation`` runs with the cache-aware OCR callable below, so
        every probe's text + metrics land in the normal content-addressed
        cache — including the winning upright frame, which makes the
        subsequent ``_ocr_jpeg`` on it a pure cache hit (the page is never
        OCR'd twice). The decision + full four-direction evidence is then
        persisted, UNLESS a probe hit an engine error — transient failures
        degrade to rotation 0 un-cached so the next run retries.
        [DECISION-pl.12]

        Public on purpose: ``table-preview --auto-rotate`` resolves its
        page through this same entry point, sharing both the mechanism and
        the sidecar cache with ingest.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        rotation_path = self._rotation_path(cache_key)
        if rotation_path.exists():
            decision: dict[str, Any] = json.loads(rotation_path.read_text(encoding="utf-8"))
            return int(decision["rotation"])

        rotation, evidence = probe_rotation(
            jpeg_bytes,
            lambda frame: self._ocr_with_scores(frame, page_num, pdf_name),
            jpeg_quality=self._jpeg_quality,
        )
        if evidence.get("engine_error"):
            logger.warning(
                "rapidocr_parser.rotation_probe_error",
                pdf=pdf_name,
                page=page_num,
            )
            return 0

        rotation_path.write_text(
            json.dumps({"rotation": rotation, **evidence}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "rapidocr_parser.rotation_decided",
            pdf=pdf_name,
            page=page_num,
            rotation=rotation,
            gated=evidence["gated"],
            cache_key=cache_key[:12],
        )
        return rotation

    def _ocr_with_scores(
        self, jpeg_bytes: bytes, page_num: int, pdf_name: str
    ) -> tuple[str, list[float]] | None:
        """Cache-aware (text, scores) OCR for one frame — the probe's ocr_fn.

        The probe needs scores (the .txt cache alone can't gate), so the
        read path requires BOTH the .txt and the .metrics sidecar; anything
        less force-runs the engine and backfills whichever artifact is
        missing — exactly the ``quality_metrics`` backfill stance
        (DECISION-cq.22: content-addressed output, never rewrite an
        existing file). Engine failure returns None (probe aborts, decision
        stays un-cached) instead of the ingest path's cached "" — caching a
        text-less verdict here could freeze a wrong rotation forever.
        """
        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        text_path = self._text_cache_path(cache_key)
        metrics_path = self._metrics_path(cache_key)
        if text_path.exists() and metrics_path.exists():
            metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
            return (
                text_path.read_text(encoding="utf-8"),
                [float(s) for s in metrics["scores"]],
            )

        try:
            engine = self._ensure_engine()
            with self._engine_call_lock:
                result = engine(jpeg_bytes)
        except Exception as exc:  # noqa: BLE001 — probe failure must not abort the parse
            logger.warning(
                "rapidocr_parser.probe_ocr_error",
                pdf=pdf_name,
                page=page_num,
                error_type=type(exc).__name__,
            )
            return None

        if result.txts is None or result.boxes is None or len(result.txts) == 0:
            text = ""
        else:
            text = _assemble_reading_order(result.boxes, result.txts)
        scores = [] if result.scores is None else [float(s) for s in result.scores]

        if not text_path.exists():
            text_path.write_text(text, encoding="utf-8")
        if not metrics_path.exists():
            self._write_metrics_sidecar(cache_key, page_num, result, text)
        return text, scores

    def _ocr_jpeg(self, jpeg_bytes: bytes, page_num: int, pdf_name: str) -> str:
        """Cache-check + local OCR for pre-rendered JPEG bytes.

        Touches no pdfium state — safe from any thread (engine call is
        serialized by the instance lock). Signature matches the other
        vendors so cli.py batch-ingest could dispatch uniformly if this
        vendor is ever wired there.

        ssQA: whenever the engine actually runs (cache miss), the per-box
        scores are persisted to the metrics sidecar alongside the .txt.
        A cache HIT deliberately does NOT backfill a missing sidecar —
        that would force an engine run inside ingest and regress its
        performance; backfill belongs to the `ocr-quality` command
        (quality_metrics below). [DECISION-cq.21]
        """
        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        cache_path = self._text_cache_path(cache_key)

        if cache_path.exists():
            logger.info(
                "rapidocr_parser.cache_hit",
                pdf=pdf_name,
                page=page_num,
                cache_key=cache_key[:12],
            )
            return cache_path.read_text(encoding="utf-8")

        # Cache miss → run local OCR. Per PDFParser contract a single page's
        # failure MUST NOT abort the batch: log + return empty, un-cached
        # (so a transient failure re-tries on the next ingest) — same stance
        # as the network vendors.
        try:
            # RapidOCR accepts raw image bytes directly (decodes internally),
            # so we never round-trip through PIL/numpy ourselves.
            engine = self._ensure_engine()
            with self._engine_call_lock:
                result = engine(jpeg_bytes)
        except Exception as exc:  # noqa: BLE001 — per-page errors must not abort batch
            # error_type only — never the message body (uniform log hygiene
            # with the other vendors, even though no secret exists here).
            logger.warning(
                "rapidocr_parser.ocr_error",
                pdf=pdf_name,
                page=page_num,
                error_type=type(exc).__name__,
            )
            return ""

        # Blank page → RapidOCR returns txts=None/boxes=None (verified
        # 2026-06-11). Normalise to "" — cached, so blank pages cost one
        # OCR pass ever, matching the sentinel handling of the LLM vendors.
        if result.txts is None or result.boxes is None or len(result.txts) == 0:
            text = ""
        else:
            text = _assemble_reading_order(result.boxes, result.txts)

        cache_path.write_text(text, encoding="utf-8")
        # ssQA: the engine ran, so the per-box scores exist exactly now —
        # persist them or lose them (a later cache hit never re-runs the
        # engine). json.dump of a few hundred floats is negligible next to
        # the ~1s OCR pass, so ingest-path cost is unchanged in substance.
        self._write_metrics_sidecar(cache_key, page_num, result, text)
        logger.info(
            "rapidocr_parser.ocr_complete",
            pdf=pdf_name,
            page=page_num,
            blocks=0 if result.txts is None else len(result.txts),
            chars=len(text),
        )
        return text

    def ocr_text_for_jpeg(self, jpeg_bytes: bytes, page_num: int, pdf_name: str) -> str:
        """Public cache-first OCR for one pre-rendered JPEG (ssTG `ocr-gallery`).

        What: thin public entry point over the existing ``_ocr_jpeg``
        cache-check + OCR path, for callers that already hold the rendered
        JPEG bytes (the gallery renders pages itself to export them as
        ``pNNNN.jpg``, so re-rendering inside ``parse()`` would be waste).

        Why a wrapper instead of exposing ``_ocr_jpeg`` directly: the
        private path assumes ``parse()`` already created the cache dir;
        this entry point owns that precondition so external callers cannot
        hit a missing-directory write error on a cold cache. Behaviour is
        otherwise identical — cache hit reads the .txt, miss runs the
        engine, writes .txt + metrics sidecar (N=2 reuse per project §5.3:
        the gallery shares this mechanism rather than copying it).
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._ocr_jpeg(jpeg_bytes, page_num, pdf_name)

    def _write_metrics_sidecar(
        self, cache_key: str, page_num: int, result: Any, text: str
    ) -> dict[str, Any]:
        """Compute + persist the quality-metrics sidecar for one engine run."""
        scores: Sequence[float] = () if result.scores is None else result.scores
        metrics = _page_metrics(page_num, scores, text)
        self._metrics_path(cache_key).write_text(
            json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
        )
        return metrics

    # ------------------------------------------------------------------
    # ssQA: per-page quality metrics (the `ocr-quality` CLI entry point)
    # ------------------------------------------------------------------

    def quality_metrics(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Per-page OCR quality metrics for ``pdf_path`` (up to ``max_pages``).

        Sidecar-first: a page whose metrics sidecar already exists is read
        from disk (no engine run). Otherwise the engine is FORCE-run — even
        when the .txt cache exists, because the text cache holds no scores —
        and the sidecar is backfilled so the next scan is free. The .txt is
        also written when (and only when) it is missing: the engine output
        is content-addressed, so this is a free, byte-equivalent backfill;
        an existing .txt is never rewritten. [DECISION-cq.22]

        This is the ONLY sidecar-backfill path — ingest never does it
        (DECISION-cq.21, zero ingest-performance regression).
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            page_count = min(total_pages, self._max_pages) if self._max_pages else total_pages
            logger.info(
                "rapidocr_parser.quality_scan_start",
                pdf=pdf_path.name,
                total_pages=total_pages,
                will_process=page_count,
                model_type=self._model_type,
            )
            records: list[dict[str, Any]] = []
            for page_idx in range(page_count):
                page_num = page_idx + 1  # 1-indexed per ParsedPage contract
                jpeg_bytes = render_page_jpeg(
                    pdf[page_idx], dpi=self._dpi, jpeg_quality=self._jpeg_quality
                )
                records.append(self._page_quality(jpeg_bytes, page_num, pdf_path.name))
            return records
        finally:
            pdf.close()

    def _page_quality(self, jpeg_bytes: bytes, page_num: int, pdf_name: str) -> dict[str, Any]:
        """Sidecar-read or engine-run quality metrics for one rendered page.

        Engine failure mirrors the ingest stance — log, return a degenerate
        record (all signals null + ``engine_error``), never abort the scan,
        never cache — so a transient failure retries on the next scan.
        """
        cache_key = hashlib.sha256(jpeg_bytes).hexdigest()
        metrics_path = self._metrics_path(cache_key)
        if metrics_path.exists():
            metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
            # Report the page we are scanning NOW; the stored value is just
            # where these bytes were first seen (content-addressed sidecar).
            metrics["page_num"] = page_num
            return metrics

        try:
            engine = self._ensure_engine()
            with self._engine_call_lock:
                result = engine(jpeg_bytes)
        except Exception as exc:  # noqa: BLE001 — per-page errors must not abort the scan
            logger.warning(
                "rapidocr_parser.quality_ocr_error",
                pdf=pdf_name,
                page=page_num,
                error_type=type(exc).__name__,
            )
            degenerate = _page_metrics(page_num, (), "")
            degenerate.update({"boxes": None, "engine_error": type(exc).__name__})
            return degenerate

        if result.txts is None or result.boxes is None or len(result.txts) == 0:
            text = ""
        else:
            text = _assemble_reading_order(result.boxes, result.txts)

        text_path = self._text_cache_path(cache_key)
        if not text_path.exists():
            text_path.write_text(text, encoding="utf-8")
        return self._write_metrics_sidecar(cache_key, page_num, result, text)
