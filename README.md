# AIgentEgo

AIgentEgo is a local-first agent runtime foundation for a developer machine.
It runs a small FastAPI service next to Ollama and keeps the orchestration
surface explicit: configuration, provider access, health checks, diagnostics,
request tracing, and smoke tests are all visible in the repository.

The project is intentionally narrow at this stage. MVP 0.1 proves that the
local stack can build, start, pull the configured Ollama models, answer a basic
chat request, and be validated repeatably. It is not yet a complete agent
runtime.

## MVP 0.1 Status

Current Milestone (X): MVP 0.1 finalization.

Included in MVP 0.1:

- Python package skeleton
- Environment-based runtime settings
- Docker application image
- Docker Compose Ollama stack
- Automatic Ollama model pull
- FastAPI runtime API
- `/health`, `/diagnostics`, and `/chat`
- Request ids and basic request logging
- Docker and Compose smoke tests
- Makefile development shortcuts

## Prerequisites

- Python 3.12
- Docker with Docker Compose v2
- `make`
- Enough disk space for the configured Ollama models

For GPU mode, the host also needs a working NVIDIA container runtime. CPU mode
is the default and does not require GPU support.

## CPU Quickstart

Start the local stack:

```sh
make up
```

The default stack builds the `agent-api` image, starts Ollama, pulls the
configured chat and embedding models, and exposes the API at
`http://localhost:8080`.

Stop the stack:

```sh
make down
```

Follow logs:

```sh
make logs
```

## Optional GPU Quickstart

Start the stack with the NVIDIA GPU Compose override:

```sh
make up-gpu
```

This only changes the Ollama service GPU reservation. The rest of the stack is
the same as the CPU path.

## Model Defaults

The defaults are chosen to stay practical on smaller local machines:

- `OLLAMA_CHAT_MODEL=llama3.2:3b`
- `OLLAMA_EMBED_MODEL=nomic-embed-text`

Runtime settings can be supplied through environment variables or a local
`.env` file. `.env` is intentionally ignored by Git; use `.env.example` as the
reference.

## Available Endpoints

### `GET /health`

Returns API status, Ollama reachability, and the configured chat model.

```sh
curl http://localhost:8080/health
```

Example shape:

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "chat_model": "llama3.2:3b"
}
```

### `GET /diagnostics`

Returns non-secret runtime diagnostics.

```sh
curl http://localhost:8080/diagnostics
```

Example shape:

```json
{
  "status": "ok",
  "package_version": "0.1.9",
  "llm_provider": "ollama",
  "ollama_base_url": "http://ollama:11434",
  "chat_model": "llama3.2:3b",
  "embedding_model": "nomic-embed-text",
  "ollama_reachable": true
}
```

### `POST /chat`

Accepts one user message and returns one non-streaming model response. Tool
calling, memory, and conversation persistence are not implemented yet.

```sh
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: local-test-1' \
  -d '{"message":"Reply with one short sentence."}'
```

Example shape:

```json
{
  "request_id": "local-test-1",
  "model": "llama3.2:3b",
  "message": "Local agents should stay small, observable, and easy to inspect."
}
```

Every API response includes an `X-Request-ID` header. If a client sends that
header, AIgentEgo preserves it in the response and in request logs.

## Validation

Run the core local validation suite:

```sh
make check
```

Run Docker image verification:

```sh
make docker-build-check
```

Validate Compose configuration:

```sh
make compose-config
```

Run the full Compose smoke test:

```sh
make compose-smoke
```

The Compose smoke test starts an isolated stack, waits for Ollama, pulls the
configured models, starts the API, checks `/health` and `/diagnostics`, sends a
real `/chat` request, and then cleans up its containers.

## Current Limitations

- `/chat` supports one public user message per request and non-streaming
  responses only.
- There is no tool registry, tool calling, memory store, notes search, calendar
  adapter, database, sandbox, or agent loop yet.
- Ollama model pull and first-response time depend on local network, disk, CPU,
  RAM, and GPU availability.
- Small laptops or machines with about 4 GB VRAM should keep the default
  `llama3.2:3b` chat model unless they have confirmed capacity for larger
  models.

## Next Direction

The next sprint should build on the validated local runtime by adding the next
minimal agent capability, while preserving the same boundaries: no hidden
services, no undocumented runtime state, and tests that do not require live
infrastructure unless they are explicitly smoke tests.
