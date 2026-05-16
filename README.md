# AIgentEgo

AIgentEgo is the starting point for a local agent runtime: a small,
inspectable service intended to run on a developer machine with Ollama behind
it and an orchestration layer around it. The project is deliberately beginning
with repository shape and operating discipline before runtime code. That keeps
early decisions visible, makes later changes reviewable, and keeps the runtime
small while the foundations are still being built.

## Current Milestone (X)

Development task shortcuts and validation workflow.

Sprint 0.1.8 adds a minimal Makefile for the common development checks, Docker
image verification, Compose validation, smoke testing, and local stack
management.

This repository is not yet a complete agent runtime. It does not currently
include tools, persistence, agent-loop behavior, memory, notes search, calendar
integration, or sandbox execution.

## Running Locally

Start the CPU default stack:

```sh
make up
```

Use the optional NVIDIA GPU override when the host Docker runtime supports it:

```sh
make up-gpu
```

The API is exposed on port `8080` by default. Supported endpoints are:

- `GET /health`
- `GET /diagnostics`
- `POST /chat` with JSON body `{"message": "Hello"}`

Every API response includes an `X-Request-ID` header. Clients may provide the
same header on incoming requests when they need to correlate their own logs with
AIgentEgo request logs.

Stop the stack with:

```sh
make down
```

Follow stack logs with:

```sh
make logs
```

## Development Workflow

Run the core local validation suite:

```sh
make check
```

Additional Docker and Compose checks are available as explicit targets:

```sh
make docker-build-check
make compose-config
make compose-smoke
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
