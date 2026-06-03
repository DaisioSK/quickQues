"""Shared ``claude -p`` subprocess runner for image-reading prompts.

Both ``ClaudeCliVisionParser`` (OCR) and ``ClaudeCliVisionCaptioner``
(E11, drawing captions) drive the ``claude`` CLI the same way: render a
page JPEG into a whitelisted dir, then ask Claude Code to Read it and
return text. The argv + JSON-envelope handling is identical; per
project_guideline.md §5 (N=2) it lives here once instead of being copied
into each impl.

Uses the caller's existing ``claude login`` OAuth — this module reads,
stores, and transmits NO API key. Token usage counts against the user's
Claude Code subscription quota.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_claude_read_image(
    *,
    claude_path: str,
    render_dir: Path,
    prompt: str,
    model: str,
    timeout_s: int,
) -> dict[str, Any]:
    """Invoke ``claude -p`` once and return the parsed JSON envelope.

    The prompt must itself instruct Claude to Read an image already
    written under ``render_dir`` (the dir is whitelisted via --add-dir).

    Returns the full parsed ``--output-format json`` envelope (caller
    pulls ``result`` and ``usage`` from it). Raises RuntimeError on a
    non-zero exit, non-JSON stdout, or an ``is_error`` envelope — callers
    catch broadly and degrade per their Protocol contract.
    """
    cmd = [
        claude_path,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--allowedTools",
        "Read",
        "--add-dir",
        str(render_dir.resolve()),
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--disable-slash-commands",
    ]

    # noqa rationale: argv is fully program-constructed (no shell=True, no
    # string concat of untrusted input). Binary path resolved by caller.
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude CLI exit {result.returncode}")

    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"claude CLI bad JSON: {result.stdout[:120]}") from exc

    if data.get("is_error"):
        raise RuntimeError(f"claude CLI api error: {data.get('subtype')}")

    return data


def run_claude_text(
    *,
    claude_path: str,
    prompt: str,
    model: str,
    timeout_s: int,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Invoke ``claude -p`` for a TEXT-only prompt (no image / no tools).

    Hardened the same way ClaudeCliAnswerer hardens its call (--tools "" so
    no agent behaviour, settings/sessions disabled). Optionally REPLACES the
    Claude Code system prompt via ``system_prompt``. Returns the parsed
    ``--output-format json`` envelope (caller pulls ``result`` / ``usage``).
    Raises RuntimeError on non-zero exit, non-JSON stdout, or ``is_error``.

    Second text caller after ClaudeCliAnswerer (Enhancement E12 judge) →
    extracted here per project_guideline.md §5 (N=2). FORESHADOW-e12.judge.1:
    ClaudeCliAnswerer still carries its own inline copy; migrating it to this
    runner is a separate refactor (it lives on the live /ask path, kept out
    of this eval-only sub-sprint's blast radius).
    """
    cmd = [claude_path, "-p", prompt, "--model", model, "--output-format", "json"]
    if system_prompt is not None:
        # REPLACE (not append) Claude Code's default system prompt.
        cmd += ["--system-prompt", system_prompt]
    cmd += [
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--tools",
        "",
        "--disable-slash-commands",
    ]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude CLI exit {result.returncode}")

    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"claude CLI bad JSON: {result.stdout[:120]}") from exc

    if data.get("is_error"):
        raise RuntimeError(f"claude CLI api error: {data.get('subtype')}")

    return data
