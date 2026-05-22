# AIgentEgo

AIgentEgo is a local-first agent runtime foundation for a developer machine.
It runs a small FastAPI service against a provider-neutral LLM boundary, with
Ollama as the default backend and optional llama.cpp compatibility behind the
same `LlmProvider` interface. The orchestration surface stays explicit:
configuration, provider access, health checks, diagnostics, manual deterministic
tools, structured ToolCall handling, request tracing, logging, and smoke tests
are all visible in the repository.

The project is still intentionally narrow. MVP 0.2 proves that the local stack
can build, start, pull the configured Ollama models, answer a basic chat
request, expose deterministic tools, execute an explicitly requested calculator
tool, and be validated repeatably. MVP 0.3 adds provider-neutral structured
ToolCall handling on top of that deterministic tool substrate. MVP 0.3.7 adds
optional llama.cpp backend compatibility for that provider-neutral boundary. It
is not yet a complete agent runtime.

## MVP 0.3.7 Status

Current Milestone (X): MVP 0.3.7 llama.cpp backend compatibility.

Included through MVP 0.3 and the current MVP 0.3.7 bridge work:

- Python package skeleton
- Environment-based runtime settings
- Docker application image
- Docker Compose Ollama stack
- Automatic Ollama model pull
- FastAPI runtime API
- `/health`, `/diagnostics`, and `/chat`
- Provider-neutral LLM models with Ollama and llama.cpp provider adapters
- Ollama as the default LLM backend
- Optional llama.cpp backend selection through `LLM_BACKEND=llamacpp`
- Deterministic tool contracts and normalized tool errors
- Tool registry and deterministic tool executor
- Calculator tool for safe arithmetic expressions
- Manual deterministic tool API through `/tools` and `/tools/execute`
- Model-facing serialization for registered tool definitions
- Strict structured ToolCall output contracts
- Parser and validator for model-produced ToolCall arrays
- Bounded repair decisions and safe-failure representation for invalid output
- Single-step structured ToolCall flow through `LlmProvider`
- Final answer synthesis from deterministic tool results or safe-failure context
- Request ids, basic request logging, and tool execution logging
- Docker and Compose smoke tests
- Makefile development shortcuts

The public tool API still invokes tools manually through explicit REST calls.
MVP 0.3 adds provider-neutral library/runtime support for single-step
LLM-requested ToolCalls, not a public agent endpoint or a multi-step loop.

In the 0.3 flow:

1. Registered deterministic tool definitions are serialized for model context.
2. Model output must be strict structured JSON with a `tool_calls` array.
3. Model-produced tool calls are parsed and validated before execution.
4. Unknown or malformed tool calls fail safely or produce bounded repair decisions.
5. Valid calls execute through the deterministic `ToolExecutor`.
6. Final answer synthesis can use deterministic tool results.

The LLM does not directly execute tools. `ToolExecutor` remains the only
execution mechanism. This is single-step structured tool calling, not a
multi-step agent loop.

MVP 0.3.7 keeps Ollama as the default backend and adds `llamacpp` as an
optional backend identifier. Application code still depends on `LlmProvider`;
backend selection remains centralized in the provider factory, and
llama.cpp-specific HTTP payloads stay inside `LlamaCppProvider`.

## Roadmap

The high-level 0.x MVP roadmap is tracked in
[docs/ROADMAP.md](docs/ROADMAP.md). It separates completed behavior from
planned future capabilities such as the agent loop, memory, RAG, filesystem
access, calendar integration, sandboxing, streaming, and CLI support.

## Prerequisites

- Python 3.12
- Docker with Docker Compose v2
- `make`
- Enough disk space for the configured Ollama models when using the default
  Compose stack

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

## LLM Backend Configuration

The defaults are chosen to stay practical on smaller local machines:

- `LLM_BACKEND=ollama`
- `LLM_BASE_URL=http://ollama:11434`
- `CHAT_MODEL=llama3.2:3b`
- `EMBEDDING_MODEL=nomic-embed-text`

Runtime settings can be supplied through environment variables or a local
`.env` file. `.env` is intentionally ignored by Git; use `.env.example` as the
reference.

Supported backend identifiers are:

- `LLM_BACKEND=ollama` for the default Ollama provider
- `LLM_BACKEND=llamacpp` for the optional llama.cpp provider

The canonical llama.cpp value is `llamacpp`. Alias spellings such as
`llama.cpp` and `llama_cpp` are not supported. The legacy `OLLAMA_BASE_URL`,
`OLLAMA_CHAT_MODEL`, and `OLLAMA_EMBED_MODEL` names remain compatibility
aliases for existing Ollama environments, but new configuration should use the
provider-neutral names above.

### Optional llama.cpp Backend

The `llamacpp` backend expects a separately managed llama.cpp `llama-server`
that exposes the OpenAI-compatible endpoints used by `LlamaCppProvider`:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Example host-run API configuration:

```sh
LLM_BACKEND=llamacpp
LLM_BASE_URL=http://localhost:8081
CHAT_MODEL=<model exposed by llama-server>
EMBEDDING_MODEL=<embedding model exposed by llama-server>
```

Embeddings are expected to work only when the selected llama.cpp server exposes
`POST /v1/embeddings` and the configured `EMBEDDING_MODEL` is valid for that
server. The default Docker Compose stack remains Ollama-first and does not
start or manage a llama.cpp service. If the API runs inside Docker, set
`LLM_BASE_URL` to an address that is reachable from the `agent-api` container.

## Available Endpoints

Every API response includes an `X-Request-ID` header. If a client sends that
header, AIgentEgo preserves it in the response and in request logs.

### `GET /health`

Returns API status, provider reachability, and the configured chat model.

```sh
curl http://localhost:8080/health
```

Example shape:

```json
{
  "status": "ok",
  "provider_reachable": true,
  "chat_model": "llama3.2:3b",
  "ollama_reachable": true
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
  "package_version": "0.3.0",
  "llm_backend": "ollama",
  "llm_provider": "ollama",
  "provider_base_url": "http://ollama:11434",
  "chat_model": "llama3.2:3b",
  "embedding_model": "nomic-embed-text",
  "provider_reachable": true,
  "ollama_base_url": "http://ollama:11434",
  "ollama_reachable": true
}
```

The `ollama_*` diagnostics fields are retained as compatibility fields while
the provider-neutral fields are the canonical shape going forward.

### `POST /chat`

Accepts one user message and returns one non-streaming model response. The chat
endpoint talks to the configured LLM provider; it does not invoke the
structured ToolCall flow or execute deterministic tools.

```sh
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: local-chat-1' \
  -d '{"message":"Reply with one short sentence."}'
```

Example shape:

```json
{
  "request_id": "local-chat-1",
  "model": "llama3.2:3b",
  "message": "Local agents should stay small, observable, and easy to inspect."
}
```

### `GET /tools`

Lists deterministic tools registered by the runtime.

```sh
curl http://localhost:8080/tools
```

Example shape:

```json
{
  "tools": [
    {
      "name": "calculator",
      "description": "Evaluate a safe arithmetic expression.",
      "parameters_schema": {
        "type": "object",
        "required": ["expression"],
        "properties": {
          "expression": {
            "type": "string",
            "description": "Arithmetic expression to evaluate."
          }
        },
        "additionalProperties": false
      }
    }
  ]
}
```

### `POST /tools/execute`

Executes one explicitly requested deterministic tool call. This endpoint is
manual and API-driven: the client chooses the tool name and supplies arguments.
The LLM is not involved in tool selection or execution.

```sh
curl -X POST http://localhost:8080/tools/execute \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: local-tool-1' \
  -d '{"tool_name":"calculator","arguments":{"expression":"12 * 31"}}'
```

Example shape:

```json
{
  "request_id": "local-tool-1",
  "tool_name": "calculator",
  "success": true,
  "result": {
    "value": 372
  },
  "error": null
}
```

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
real `/chat` request, verifies `GET /tools`, manually executes the calculator
through `POST /tools/execute`, and then cleans up its containers.

### Optional llama.cpp Manual Validation

llama.cpp validation is optional and local. It is not part of the default
`make compose-smoke` path.

1. Start `llama-server` separately with a chat model and, if needed, an
   embedding model.
2. Confirm the server is reachable:

   ```sh
   curl http://localhost:8081/health
   curl http://localhost:8081/v1/models
   ```

3. Start the API with llama.cpp settings:

   ```sh
   LLM_BACKEND=llamacpp \
   LLM_BASE_URL=http://localhost:8081 \
   CHAT_MODEL=<model exposed by llama-server> \
   EMBEDDING_MODEL=<embedding model exposed by llama-server> \
   python -m aigentego.main
   ```

4. Check the regular provider-neutral endpoints:

   ```sh
   curl http://localhost:8080/health
   curl http://localhost:8080/diagnostics
   curl -X POST http://localhost:8080/chat \
     -H 'Content-Type: application/json' \
     -d '{"message":"Reply with one short sentence."}'
   ```

Structured ToolCall compatibility for llama.cpp is covered by deterministic
mocked tests. The LLM still requests tools only through parsed and validated
structured JSON, and valid calls still execute only through `ToolExecutor`.

## Current Limitations

- `/chat` supports one public user message per request and non-streaming
  responses only.
- Public deterministic tool endpoints are manual/API-driven only.
- Structured ToolCall support is provider-neutral and single-step; no public
  `/agent/run` endpoint or multi-step agent loop exists yet.
- There is no persistent memory store, notes search, read-only filesystem tool,
  calendar adapter, database integration, sandbox, CLI, MCP, or streaming
  support.
- llama.cpp is optional and externally managed; the default Compose stack does
  not include a llama.cpp service or automatic GGUF model download.
- Ollama model pull and first-response time depend on local network, disk, CPU,
  RAM, and GPU availability.
- Small laptops or machines with about 4 GB VRAM should keep the default
  `llama3.2:3b` chat model unless they have confirmed capacity for larger
  models.

## Next Direction

The active bridge milestone is llama.cpp backend compatibility. After closure,
the next planned runtime milestone is Agent Loop v1: a bounded multi-step loop
with explicit limits and observations.
