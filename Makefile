.DEFAULT_GOAL := help

PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: help
help:
	@printf '%s\n' \
		'Available targets:' \
		'  help                 Show this help message' \
		'  test                 Run the full test suite' \
		'  test-unit            Run unit tests' \
		'  lint                 Run Ruff lint checks' \
		'  typecheck            Run mypy type checks' \
		'  check                Run lint, typecheck, and tests' \
		'  docker-build-check   Build and verify the Docker image' \
		'  compose-config       Validate Docker Compose configuration' \
		'  compose-smoke        Run the Docker Compose smoke test' \
		'  up                   Start the default Compose stack' \
		'  up-gpu               Start the Compose stack with NVIDIA GPU support' \
		'  down                 Stop the Compose stack' \
		'  logs                 Follow Compose logs'

.PHONY: test
test:
	$(PYTHON) -m pytest

.PHONY: test-unit
test-unit:
	$(PYTHON) -m pytest

.PHONY: lint
lint:
	$(PYTHON) -m ruff check .

.PHONY: typecheck
typecheck:
	$(PYTHON) -m mypy

.PHONY: check
check: lint typecheck test

.PHONY: docker-build-check
docker-build-check:
	scripts/docker_build_check.sh

.PHONY: compose-config
compose-config:
	$(COMPOSE) config

.PHONY: compose-smoke
compose-smoke:
	scripts/compose_smoke_test.sh

.PHONY: up
up:
	$(COMPOSE) up --build

.PHONY: up-gpu
up-gpu:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up --build

.PHONY: down
down:
	$(COMPOSE) down --remove-orphans

.PHONY: logs
logs:
	$(COMPOSE) logs -f
