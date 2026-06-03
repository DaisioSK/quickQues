"""Load DomainProfile definitions from ``profiles/<name>.yaml`` (Phase 7).

`load_profile(name)` reads + validates a YAML profile into the Layer-0
`DomainProfile`/`StructureSpec` dataclasses. Profiles live at the repo
root under `profiles/` (override with `JCONTRACT_PROFILES_DIR` for tests
or alternate deployments). Results are cached — profiles are static per
process.

Adding a new domain = drop a `profiles/<name>.yaml`; no code change here.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from jcontract.interfaces import DomainProfile, RefRule, StructureSpec

# Repo-root profiles dir: this file is src/jcontract/impls/…, so parents[3]
# is the repo root. Overridable via env for tests / alt deployments.
_DEFAULT_PROFILES_DIR = Path(__file__).resolve().parents[3] / "profiles"


def _profiles_dir() -> Path:
    override = os.environ.get("JCONTRACT_PROFILES_DIR")
    return Path(override) if override else _DEFAULT_PROFILES_DIR


def available_profiles() -> list[str]:
    """Return the sorted names of all `profiles/*.yaml` (without extension)."""
    d = _profiles_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def _require(d: dict[str, Any], key: str, name: str) -> Any:
    if key not in d:
        raise ValueError(f"profile '{name}' missing required key: {key!r}")
    return d[key]


def _parse_structure(raw: dict[str, Any], name: str) -> StructureSpec:
    rules_raw = raw.get("ref_rules") or []
    if not isinstance(rules_raw, list):
        raise ValueError(f"profile '{name}': structure.ref_rules must be a list")
    ref_rules = tuple(
        RefRule(pattern=str(r["pattern"]), target_field=str(r["target_field"])) for r in rules_raw
    )
    qa = raw.get("qa_block_pattern")
    section = raw.get("section_header_pattern")
    clause = raw.get("clause_header_pattern")
    return StructureSpec(
        qa_block_pattern=None if qa is None else str(qa),
        ref_rules=ref_rules,
        section_header_pattern=None if section is None else str(section),
        clause_header_pattern=None if clause is None else str(clause),
    )


@cache
def load_profile(name: str) -> DomainProfile:
    """Load + validate ``profiles/<name>.yaml`` into a DomainProfile.

    Raises ValueError on an unknown name or a malformed profile.
    """
    path = _profiles_dir() / f"{name}.yaml"
    if not path.exists():
        raise ValueError(
            f"Unknown domain profile {name!r}. Available: {available_profiles()} "
            f"(looked in {_profiles_dir()})."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"profile '{name}': top-level YAML must be a mapping")

    structure_raw = _require(data, "structure", name)
    if not isinstance(structure_raw, dict):
        raise ValueError(f"profile '{name}': 'structure' must be a mapping")

    suggested = data.get("suggested_questions") or []
    if not isinstance(suggested, list):
        raise ValueError(f"profile '{name}': suggested_questions must be a list")

    return DomainProfile(
        name=str(_require(data, "name", name)),
        answer_framing=str(_require(data, "answer_framing", name)),
        ocr_text_prompt=str(_require(data, "ocr_text_prompt", name)),
        ocr_drawing_prompt=str(_require(data, "ocr_drawing_prompt", name)),
        caption_prompt=str(_require(data, "caption_prompt", name)),
        structure=_parse_structure(structure_raw, name),
        suggested_questions=tuple(str(q) for q in suggested),
    )
