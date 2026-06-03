# ============================================================
# j-contract — developer command shortcuts
# ============================================================
# Self-documenting Makefile: run `make` (or `make help`) to list targets.
# All commands mirror docs/README.md + scripts/check.sh; this file is just
# a faster front door, it adds no new behaviour.
#
# Common overridable variables (pass on the command line, e.g. make PORT=9000 backend):
#   ANSWERER  answerer backend: claude-api | claude-cli | codex-cli   (default: claude-cli)
#   PARSER    pdf parser: pypdf | claude-vision | claude-cli-vision | deepseek-v4
#   HOST/PORT backend bind host/port            (default: 0.0.0.0 / 8000)
#   PDF       path to a PDF for the `ingest` target
#   Q         query string for the `search` target
# ============================================================

# --- tunables -------------------------------------------------
ANSWERER ?= claude-cli
HOST     ?= 0.0.0.0
PORT     ?= 8000
PARSER   ?= claude-cli-vision
MAX_PAGES ?=
PDF      ?=
Q        ?=
K        ?= 5
N        ?= 20

# uvicorn app path is fixed.
APP := jcontract.api.main:app

# Compose command differs by install: v2 plugin ("docker compose") vs v1 ("docker-compose").
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.DEFAULT_GOAL := help
.PHONY: help install env hooks qdrant-up qdrant-down qdrant-logs \
        dev backend frontend health \
        ingest search chunks refs evaluate \
        check check-web check-all lint format format-fix typecheck test test-web

# --- help (default) -------------------------------------------
help: ## Show this help
	@echo "j-contract — make targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make dev                                       # qdrant + backend + frontend, one command"
	@echo "  make ingest PDF='input-docs/Contract DEMO(1of9) TQA.pdf'"
	@echo "  make search Q='防水谁负责'"
	@echo "  make backend ANSWERER=claude-api PORT=9000"

# --- setup ----------------------------------------------------
install: ## Install all deps (python venv + frontend) and git hooks
	uv sync
	cd web && npm install
	bash scripts/install-hooks.sh

env: ## Create .env from template if it does not exist
	@test -f .env && echo ".env already exists — leaving it untouched" \
		|| (cp .env.example .env && echo "created .env — fill in your keys (optional: claude-cli needs none)")

hooks: ## Install the pre-commit secret guard
	bash scripts/install-hooks.sh

# --- infra ----------------------------------------------------
qdrant-up: ## Start the Qdrant vector DB (docker) — no-op if already up
	@curl -sf -m2 localhost:6333/healthz >/dev/null 2>&1 \
		&& echo "==> qdrant already up" \
		|| $(COMPOSE) up -d qdrant

qdrant-down: ## Stop the Qdrant vector DB
	$(COMPOSE) stop qdrant

qdrant-logs: ## Tail Qdrant logs
	$(COMPOSE) logs -f qdrant

# --- run ------------------------------------------------------
dev: qdrant-up ## ONE COMMAND: qdrant + backend + frontend (Ctrl-C stops all)
	@echo "==> backend on http://$(HOST):$(PORT)  |  frontend on http://localhost:3000"
	@echo "==> answerer backend = $(ANSWERER)   (Ctrl-C to stop both)"
	@trap 'kill 0' EXIT INT TERM; \
	JCONTRACT_ANSWERER_BACKEND=$(ANSWERER) uv run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload & \
	(cd web && npm run dev) & \
	wait

backend: qdrant-up ## Start the FastAPI backend only (with --reload)
	JCONTRACT_ANSWERER_BACKEND=$(ANSWERER) uv run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

frontend: ## Start the Next.js frontend only
	cd web && npm run dev

health: ## Curl the backend liveness + index count
	curl -s http://localhost:$(PORT)/healthz | python3 -m json.tool

# --- data / retrieval -----------------------------------------
ingest: ## Ingest a PDF into the index — needs PDF=... (PARSER/MAX_PAGES overridable)
	@test -n "$(PDF)" || (echo "ERROR: pass PDF=... e.g. make ingest PDF='input-docs/Contract DEMO(1of9) TQA.pdf'"; exit 1)
	uv run jcontract ingest "$(PDF)" --parser $(PARSER) $(if $(MAX_PAGES),--max-pages $(MAX_PAGES),)

search: ## Retrieval debug (bypasses LLM) — needs Q=... (K overridable)
	@test -n "$(Q)" || (echo "ERROR: pass Q=... e.g. make search Q='防水谁负责'"; exit 1)
	uv run jcontract search "$(Q)" --k $(K)

chunks: ## Show what is currently indexed (N overridable)
	uv run jcontract show-chunks --n $(N)

refs: ## RefGraph lookup — make refs TYPE=drawing VALUE=T/PRJ/...
	@test -n "$(TYPE)" -a -n "$(VALUE)" || (echo "ERROR: pass TYPE=... VALUE=..."; exit 1)
	uv run jcontract refs $(TYPE) "$(VALUE)"

evaluate: ## Run the golden eval set (ANSWERER overridable)
	uv run jcontract evaluate --answerer $(ANSWERER)

# --- quality gates --------------------------------------------
check: ## Python three-piece gate (ruff + mypy + pytest)
	bash scripts/check.sh

check-web: ## Frontend gate (eslint + tsc + vitest)
	bash web/scripts/check.sh

check-all: check check-web ## Run BOTH gates (run before merging)

lint: ## ruff check .
	uv run ruff check .

format: ## ruff format --check .
	uv run ruff format --check .

format-fix: ## ruff format . (apply)
	uv run ruff format .

typecheck: ## mypy .
	uv run mypy .

test: ## pytest
	uv run pytest

test-web: ## frontend vitest
	cd web && npm run test
