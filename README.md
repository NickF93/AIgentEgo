# AIgentEgo

AIgentEgo is the starting point for a local agent runtime: a small,
inspectable service intended to run on a developer machine with Ollama behind
it and an orchestration layer around it. The project is deliberately beginning
with repository shape and operating discipline before runtime code. That keeps
early decisions visible, makes later changes reviewable, and keeps the runtime
small while the foundations are still being built.

## Current Milestone (X)

Minimal logging and request tracing.

Sprint 0.1.7 adds basic standard-library logging, per-request identifiers, the
`X-Request-ID` response header, and minimal request timing logs for the existing
FastAPI service.

This repository is not yet a complete agent runtime. It does not currently
include tools, persistence, agent-loop behavior, memory, notes search, calendar
integration, or sandbox execution.

## Running Locally

Start the CPU default stack:

```sh
docker compose up --build
```

Use the optional NVIDIA GPU override when the host Docker runtime supports it:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The API is exposed on port `8080` by default. Supported endpoints are:

- `GET /health`
- `GET /diagnostics`
- `POST /chat` with JSON body `{"message": "Hello"}`

Every API response includes an `X-Request-ID` header. Clients may provide the
same header on incoming requests when they need to correlate their own logs with
AIgentEgo request logs.

Run the Compose smoke test with:

```sh
scripts/compose_smoke_test.sh
```

## Planned MVP 0.1 Scope

- Dockerized FastAPI service
- Ollama backend
- Automatic model pull
- `/health`, `/diagnostics`, and `/chat` endpoints
- Smoke tests

## Future Roadmap

- Tool registry
- Read-only filesystem tools
- Local notes search
- Memory store
- Calendar adapter
- Sandboxed Python execution
