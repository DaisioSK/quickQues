# syntax=docker/dockerfile:1.7
# ============================================================
# j-contract Dockerfile — multi-stage with uv
# ============================================================
# Stage 1 (builder): uv resolves and installs deps into /opt/venv.
# Stage 2 (runtime): copies /opt/venv + src/ into a slim image.
#
# Placeholder CMD — replaced when FastAPI is wired in Phase 5.
# FORESHADOW-0.3.1: GPU support is NOT included here. When bge-m3 /
# bge-reranker run inside this container (Phase 2 S2.2+), revisit base
# image (e.g. nvidia/cuda runtime) + docker-compose deploy.resources.
# ============================================================

# ----- Stage 1: builder -----
FROM python:3.12-slim AS builder

# Reproducible uv binary pinned alongside the project's uv lockfile generator.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install deps first (cache-friendly): copy only the manifests.
COPY pyproject.toml uv.lock ./

# README required by the build because pyproject points to it.
COPY README.md ./

COPY src/ ./src/

# Sync without dev tooling for the runtime image.
# Note: cache mount (--mount=type=cache) requires BuildKit. Add back when
# we move to docker compose v2 (FORESHADOW-0.3.2 in docker-compose.yml).
RUN uv sync --frozen --no-dev

# ----- Stage 2: runtime -----
FROM python:3.12-slim AS runtime

# Minimal OS packages. Phase 1+ will add poppler-utils / tesseract when
# OCR is wired in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for the app.
RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ /app/src/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
USER app

# Placeholder — replaced by FastAPI/uvicorn entry in Phase 5.
CMD ["python", "-c", "import jcontract; print(f'jcontract {jcontract.__version__} ready')"]
