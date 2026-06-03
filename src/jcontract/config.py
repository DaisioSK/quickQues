"""Centralized configuration and secret loading.

Per docs/project_guideline.md §6.1 and dev-contract/21-domain-security.md:
- All secrets come from environment variables (loaded from .env in dev).
- No hardcoded defaults that are secrets.
- Required keys validated at startup; failure is loud and immediate.
- Never log values; only log key names when reporting missing config.

Phase 1 S1.2 will start consuming LLAMA_CLOUD_API_KEY here.
Phase 1 S1.4 will start consuming ANTHROPIC_API_KEY here.
Each first-use Sub-sprint must upgrade to High-Risk Mode per §6.D of plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Application-level (non-secret) configuration."""

    app_port: int
    log_level: str
    qdrant_url: str


def _require_env(key: str) -> str:
    """Fetch a required environment variable.

    Raises a clear error mentioning only the key name, never the value.
    """
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"Required environment variable not set: {key}. "
            f"Copy .env.example to .env and fill values."
        )
    return value


def _optional_env(key: str, default: str) -> str:
    """Fetch an optional environment variable with a non-secret default."""
    return os.environ.get(key, default)


def load_app_config() -> AppConfig:
    """Load app-level config. Safe to call at startup; reads no secrets."""
    return AppConfig(
        app_port=int(_optional_env("APP_PORT", "8000")),
        log_level=_optional_env("LOG_LEVEL", "INFO"),
        qdrant_url=_optional_env("QDRANT_URL", "http://localhost:6333"),
    )


# Secret accessors — defined as functions, not module-level constants, so
# that .env doesn't have to be present at import time.


def get_anthropic_api_key() -> str:
    """Anthropic / Claude API key. Required for Vision + Answerer."""
    return _require_env("ANTHROPIC_API_KEY")


def get_deepseek_api_key() -> str:
    """DeepSeek API key. Required for Answerer fallback."""
    return _require_env("DEEPSEEK_API_KEY")


def get_llama_cloud_api_key() -> str:
    """LlamaParse (LlamaIndex Cloud) API key. Required for PDF parsing."""
    return _require_env("LLAMA_CLOUD_API_KEY")


def get_qdrant_api_key() -> str | None:
    """Qdrant API key — optional for self-hosted local dev."""
    return os.environ.get("QDRANT_API_KEY") or None
