# AIgentEgo

AIgentEgo is the starting point for a local agent runtime: a small,
inspectable service intended to run on a developer machine with Ollama behind
it and an orchestration layer around it. The project is deliberately beginning
with repository shape and operating discipline before runtime code. That keeps
early decisions visible, makes later changes reviewable, and avoids pretending
there is an agent system before the foundations exist.

## Current Milestone (X)

Python package skeleton.

Sprint 0.1.1 adds the minimal `src/` package layout, an import test, and
initial Ruff, mypy, and pytest configuration.

This repository is not yet a complete agent runtime. It does not currently
include runtime application code, API endpoints, Docker configuration, model
integration, tools, persistence, or sandbox execution.

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
