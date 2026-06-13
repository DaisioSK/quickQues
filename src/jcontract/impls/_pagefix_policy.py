"""Pagefix policy — the four ssPageFix valves' thresholds + toggles as data.

What:
    ``PagefixPolicy`` is a frozen dataclass that gathers every tunable of the
    four v7 page-fix valves — orientation probe (ssRT), region assembly
    (ssGE), needs-vision classifier v2 (ssVR), DPI-escalation rescue (ssHD)
    — into ONE config object: a per-valve on/off toggle plus the numeric
    thresholds each valve reads. ``load_policy(name_or_path)`` reads a
    ``pagefix-policy.yaml`` into that dataclass, mirroring the DomainProfile
    loader (``profiles/<name>.yaml`` + ``JCONTRACT_PAGEFIX_POLICY`` override).

Why (DECISION-pm.2):
    Until now each valve's thresholds were hard-coded module constants spread
    across three files (``_page_orient`` / ``_page_classify`` /
    ``rapidocr_parser``). That made "try a different policy and see what flips"
    a source edit + reinstall, and gave the mock/trace tooling (ssMock) and a
    future real-pipeline wiring (FORESHADOW-pm.1) no single object to read.
    Thresholds are a *runtime strategy*, not toolchain config, so they live in
    the data-config layer (next to ``profiles/*.yaml``), NOT in pyproject.

Zero-behaviour-change contract:
    Every threshold field DEFAULTS to the live module constant it replaces —
    imported here, not transcribed — so the built-in default policy is byte-
    equal to the current hard-coded behaviour *by construction*: if a constant
    moves, this default moves with it (a regression test pins the equality).
    The decision functions keep those constants as their own kwarg defaults,
    so nothing reads this policy unless a caller explicitly injects it. The
    valve TOGGLES carry the DECISION-pm.3 decoupled defaults
    (rotate/rescue=ON, regions/v2=OFF) — but those are POLICY defaults consumed
    by the mock/trace surface only; they do NOT change the ingest pipeline's
    own CLI-flag defaults (all opt-in, unchanged). Wiring the real pipeline to
    read this policy is FORESHADOW-pm.1 (not this sub-sprint). [DECISION-pm.10]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any

import yaml

# Field defaults are the LIVE constants, imported (not copied), so the
# built-in default policy tracks the source of truth byte-for-byte.
from jcontract.impls._page_classify import (
    V2_FILLED_DARK_RATIO,
    V2_FRAGMENT_BOX_FRAC,
    V2_SPARSE_COVER,
    V2_SPARSE_DARK,
)
from jcontract.impls._page_orient import GATE_MIN_SCORE, IMPROVEMENT_FACTOR
from jcontract.impls.rapidocr_parser import RESCUE_DPI, RESCUE_MIN_SCORE

# Repo-root profiles dir (this file is src/jcontract/impls/…, parents[3] is
# the repo root) — the same convention DomainProfile uses. Override via
# JCONTRACT_PAGEFIX_POLICY for tests / alternate deployments. [DECISION-pm.10]
_DEFAULT_PROFILES_DIR = Path(__file__).resolve().parents[3] / "profiles"
POLICY_ENV_VAR = "JCONTRACT_PAGEFIX_POLICY"
# The built-in framework default档 lives at this name; load_policy("default")
# (or an empty/absent override) resolves to profiles/pagefix-policy.yaml.
DEFAULT_POLICY_NAME = "pagefix-policy"


@dataclass(frozen=True)
class PagefixPolicy:
    """The four page-fix valves' toggles + thresholds, as one config object.

    Field groups:
      * Valve toggles (DECISION-pm.3 decoupled defaults): which valves the
        policy turns on. rotate/rescue default ON (low-risk, valve-gated,
        net-positive on the W6 corpus); regions/v2 default OFF (net-negative
        on this corpus — regions hurts clause pages, v2 over-sends to vision).
      * ssRT (orientation probe): gate_min_score / improvement_factor.
      * ssVR (classify v2): the four classify_page_v2 thresholds.
      * ssHD (rescue): rescue_dpi / rescue_min_score.

    Every threshold defaults to its live module constant, so the no-arg
    ``PagefixPolicy()`` is byte-equal to the current hard-coded behaviour.
    """

    # --- valve toggles (decoupled per-valve defaults, DECISION-pm.3) ---
    rotate: bool = True
    regions: bool = False
    needs_vision_v2: bool = False
    rescue: bool = True

    # --- ssRT orientation probe thresholds (_page_orient) ---
    gate_min_score: float = GATE_MIN_SCORE
    improvement_factor: float = IMPROVEMENT_FACTOR

    # --- ssVR classify_page_v2 thresholds (_page_classify) ---
    v2_sparse_cover: float = V2_SPARSE_COVER
    v2_sparse_dark: float = V2_SPARSE_DARK
    v2_fragment_box_frac: float = V2_FRAGMENT_BOX_FRAC
    v2_filled_dark_ratio: float = V2_FILLED_DARK_RATIO

    # --- ssHD DPI-escalation rescue thresholds (rapidocr_parser) ---
    rescue_dpi: int = RESCUE_DPI
    rescue_min_score: float = RESCUE_MIN_SCORE


# Field name -> (yaml section, yaml key). Keeps the dataclass flat (decision
# functions take flat kwargs) while the YAML groups by valve for readability.
_TOGGLE_FIELDS = ("rotate", "regions", "needs_vision_v2", "rescue")
_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "ssrt": ("gate_min_score", "improvement_factor"),
    "ssvr": (
        "v2_sparse_cover",
        "v2_sparse_dark",
        "v2_fragment_box_frac",
        "v2_filled_dark_ratio",
    ),
    "sshd": ("rescue_dpi", "rescue_min_score"),
}
_FIELD_TYPES = {f.name: f.type for f in fields(PagefixPolicy)}


def _policy_path(name_or_path: str) -> Path:
    """Resolve a policy name or explicit path to a YAML file.

    A value containing a path separator or ending in ``.yaml`` is treated as
    an explicit path; a bare name resolves under the (env-overridable)
    profiles dir, exactly like DomainProfile. [DECISION-pm.10]
    """
    if name_or_path.endswith(".yaml") or os.sep in name_or_path or "/" in name_or_path:
        return Path(name_or_path)
    override = os.environ.get(POLICY_ENV_VAR)
    base = Path(override) if override else _DEFAULT_PROFILES_DIR
    return base / f"{name_or_path}.yaml"


def _coerce(field_name: str, value: Any) -> Any:
    """Coerce a YAML scalar to the dataclass field's type (bool/int/float)."""
    declared = _FIELD_TYPES[field_name]
    if declared == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"pagefix policy: {field_name!r} must be a bool, got {value!r}")
        return value
    if declared == "int":
        return int(value)
    return float(value)


@cache
def load_policy(name_or_path: str = DEFAULT_POLICY_NAME) -> PagefixPolicy:
    """Load a ``pagefix-policy.yaml`` into a :class:`PagefixPolicy`.

    With no argument (or a missing file) returns the built-in default policy
    == the live module constants + the DECISION-pm.3 toggle defaults. A
    partial YAML overrides only the keys it names; everything else keeps the
    built-in default, so a policy file need only state what it changes.

    Raises ``ValueError`` on a malformed file or an unknown key.
    """
    path = _policy_path(name_or_path)
    if not path.exists():
        if name_or_path == DEFAULT_POLICY_NAME:
            # No file shipped yet → the in-code defaults ARE the default
            # policy (byte-equal to current behaviour). Never raise for the
            # built-in name: the dataclass is the authoritative fallback.
            return PagefixPolicy()
        raise ValueError(
            f"Unknown pagefix policy {name_or_path!r} (looked in {path}). "
            f"Built-in default name: {DEFAULT_POLICY_NAME!r}."
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return PagefixPolicy()
    if not isinstance(data, dict):
        raise ValueError(f"pagefix policy {path}: top-level YAML must be a mapping")

    kwargs: dict[str, Any] = {}
    valves = data.get("valves") or {}
    if not isinstance(valves, dict):
        raise ValueError(f"pagefix policy {path}: 'valves' must be a mapping")
    for name in _TOGGLE_FIELDS:
        if name in valves:
            kwargs[name] = _coerce(name, valves[name])

    for section, field_names in _SECTION_FIELDS.items():
        block = data.get(section) or {}
        if not isinstance(block, dict):
            raise ValueError(f"pagefix policy {path}: {section!r} must be a mapping")
        for name in field_names:
            if name in block:
                kwargs[name] = _coerce(name, block[name])

    return PagefixPolicy(**kwargs)
