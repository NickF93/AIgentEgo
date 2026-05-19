# AIgentEgo Roadmap

AIgentEgo is being built as a local-first agent runtime foundation. The near
term goal is not to ship a broad agent framework, but to grow a small,
inspectable runtime in stable layers: local LLM access first, deterministic
tools next, then bounded agent behavior, memory, retrieval, integrations, and
developer-facing UX.

## Current Status

MVP 0.1 and MVP 0.2 are completed.

MVP 0.1 provides the local LLM runtime foundation:

- Python package
- Environment-based settings
- Docker application image
- Docker Compose stack with Ollama
- Ollama model auto-pull
- FastAPI API
- `/health`
- `/diagnostics`
- `/chat`
- Ollama HTTP client
- Request tracing
- Logging
- Smoke test
- Makefile development workflow
- README quickstart

MVP 0.2 adds the deterministic tool runtime:

- Provider-neutral tool contracts
- Normalized tool errors
- Deterministic `ToolRegistry`
- Deterministic `ToolExecutor`
- Safe arithmetic `CalculatorTool`
- `GET /tools`
- `POST /tools/execute`
- Tool execution request tracing and logging
- Smoke coverage for manual tool execution
- README documentation for explicit API-driven tools

AIgentEgo is still not a complete agent runtime. The following capabilities are
not implemented yet: LLM-selected tool calling, provider-neutral backend
factory, llama.cpp backend support, agent loop, memory, RAG, notes search, file
access, calendar integration, Python sandbox, CLI, streaming, and persisted
multi-turn conversation state.

## Roadmap Principles

- Keep each milestone narrow enough to validate locally.
- Separate deterministic runtime behavior from LLM-driven decision-making.
- Prefer explicit contracts, bounded execution, and observable behavior.
- Treat filesystem access, calendar writes, and Python execution as
  safety-sensitive capabilities.
- Document planned capabilities as planned, not as available behavior.

## Milestone Overview

| Version | Milestone name | Status | Primary capability | Expected user-visible outcome |
| --- | --- | --- | --- | --- |
| `0.1` | Local LLM runtime foundation | Completed | Local Ollama-backed API foundation | Run a local API with health, diagnostics, chat, tracing, and smoke validation |
| `0.2` | Deterministic tool runtime | Completed | Manual, deterministic tool execution | Execute registered tools through explicit API calls without LLM choice |
| `0.2.7` | Provider-neutral LLM runtime boundary | Planned | Generic LLM settings, provider factory, and diagnostics | Prepare application layers to depend on `LlmProvider`, not a concrete backend |
| `0.3` | LLM structured output to ToolCall | Planned | Model-produced structured tool calls | Let the LLM request bounded tool calls through validated structured output |
| `0.3.7` | llama.cpp backend compatibility | Planned | Additional local backend adapter and capability comparison | Compare Ollama and llama.cpp behavior before building the agent loop |
| `0.4` | Agent loop v1 | Planned | Bounded multi-step agent execution | Run a minimal inspectable agent loop with limits and observations |
| `0.5` | Persistent conversations and memory | Planned | Local persistence and conversation memory | Resume sessions and inject saved context into chat or agent runs |
| `0.6` | Notes search, read-only filesystem, and RAG v1 | Planned | Local retrieval over explicit read-only roots | Search notes/files and use retrieved snippets as grounded context |
| `0.7` | Calendar integration | Planned | Calendar query tools and adapters | Query local/fake calendars first, with approval-gated write intent later |
| `0.8` | Python sandbox | Planned | Restricted Python execution | Run approved Python snippets inside a constrained sandbox |
| `0.9` | Streaming and CLI | Planned | Streaming UX and terminal interface | Stream chat/agent events and use the runtime through a local CLI |
| `0.10` | Evaluation, hardening, and portfolio release | Planned | Quality, reproducibility, and release polish | Validate behavior, document architecture, and prepare demo-ready workflows |

## `0.1` Local LLM Runtime Foundation

Status: completed.

MVP 0.1 established the repository, package, local runtime service, Ollama
integration, request tracing, smoke validation, and README quickstart. It is a
foundation for future agent work, not a full agent runtime.

Internal sprint blocks:

- `0.1.0` = repository setup
- `0.1.1` = Python package skeleton
- `0.1.2` = settings layer
- `0.1.3` = Docker application image
- `0.1.4` = Docker Compose and Ollama stack
- `0.1.5` = Ollama HTTP client/provider
- `0.1.6` = FastAPI runtime API
- `0.1.7` = logging and request tracing
- `0.1.8` = development workflow / Makefile
- `0.1.9` = MVP documentation and closure

## `0.2` Deterministic Tool Runtime

Status: completed.

This milestone implemented deterministic tool execution without LLM
decision-making. Tools are executed manually or through deterministic API calls.
The LLM does not choose tools yet, no agent loop is introduced yet, and no
memory or RAG is introduced yet.

High-level sprint blocks:

- `0.2.0` = tool contracts and error taxonomy
- `0.2.1` = ToolRegistry
- `0.2.2` = ToolExecutor
- `0.2.3` = CalculatorTool
- `0.2.4` = manual tool execution API
- `0.2.5` = tool observability and request tracing integration
- `0.2.6` = tests, smoke coverage, and README update

## `0.2.7` Provider-Neutral LLM Runtime Boundary

Status: planned.

This bridge phase prevents the rest of AIgentEgo from becoming semantically
locked to Ollama before LLM-produced ToolCalls are introduced. Application
layers should depend on the provider-neutral `LlmProvider` protocol, not on
concrete provider adapters.

Planned boundary work:

- Introduce canonical generic settings:
  - `LLM_BACKEND`
  - `LLM_BASE_URL`
  - `CHAT_MODEL`
  - `EMBEDDING_MODEL`
  - `REQUEST_TIMEOUT_SECONDS`
- Add a provider factory that constructs the configured `LlmProvider`.
- Refactor diagnostics toward provider-neutral fields where possible.
- Preserve compatibility handling for existing Ollama-specific environment
  variables if it is needed for a smooth migration.
- Keep Ollama as the only implemented backend in this phase.
- Do not implement llama.cpp, structured ToolCalls, or the agent loop here.

Desired dependency direction:

```text
FastAPI routes / future structured ToolCall flow / future agent loop / future RAG
  -> LlmProvider protocol
  -> provider factory
  -> concrete provider adapter:
       - OllamaClient
       - future LlamaCppProvider
```

Rules for this boundary:

- Concrete providers must not depend on each other.
- Application code must not branch on Ollama vs llama.cpp except inside the
  provider factory or narrowly defined compatibility/capability code.
- Backend-specific HTTP payloads must stay inside provider adapters.
- Normalized LLM errors must remain provider-neutral.
- Documentation must continue to distinguish implemented behavior from planned
  behavior.

## `0.3` LLM Structured Output to ToolCall

Status: planned.

This milestone introduces LLM-assisted tool calling through structured output.
It is still not a full multi-step agent loop. Execution remains bounded and
inspectable, with validation between model output and tool execution. It should
build on the provider-neutral boundary so structured-output behavior is not
hardwired to one backend's HTTP payload shape.

High-level sprint blocks:

- `0.3.0` = tool schema serialization for LLM context
- `0.3.1` = structured ToolCall output contract
- `0.3.2` = parser and validator for model-produced ToolCall arrays
- `0.3.3` = retry/repair path for invalid structured output
- `0.3.4` = single-step LLM tool execution flow
- `0.3.5` = final answer synthesis after tool result
- `0.3.6` = evaluation cases for tool selection accuracy

## `0.3.7` llama.cpp Backend Compatibility

Status: planned.

This bridge phase comes after structured ToolCall support and before Agent
Loop v1. Ollama remains the default backend. A future `LlamaCppProvider` is
intended for CPU-only, low-RAM, edge, GGUF-controlled, or benchmarking-oriented
deployments.

The goal is compatibility evidence before building the full agent loop. The
runtime should compare backend behavior for chat, embeddings, structured output
reliability, error handling, and operational constraints while keeping provider
details behind the `LlmProvider` boundary.

Provider capability metadata may be introduced if real backend differences
require it:

- `supports_chat`
- `supports_embeddings`
- `supports_streaming`
- `supports_json_mode`
- `supports_schema_constrained_output`
- `supports_tool_calling_api`

Any additional capability flag should be justified by an actual backend
difference. This milestone must not introduce the agent loop, memory, RAG, CLI,
or streaming UX.

## `0.4` Agent Loop v1

Status: planned.

This is the first true minimal agent milestone. The loop must be bounded, and
safety limits such as max steps, timeout, and max tool errors are part of the
milestone. Persistent memory and RAG are not required yet.

High-level sprint blocks:

- `0.4.0` = AgentRun and AgentStep contracts
- `0.4.1` = bounded loop executor skeleton
- `0.4.2` = integrate LLM ToolCall generation into the loop
- `0.4.3` = integrate ToolExecutor observations
- `0.4.4` = final answer synthesis
- `0.4.5` = loop safety limits
- `0.4.6` = `/agent/run` API
- `0.4.7` = tests and smoke coverage for bounded agent runs

## `0.5` Persistent Conversations and Memory

Status: planned.

This milestone enables persisted multi-turn interaction. SQLite is the
preferred first storage backend for local-first MVP work. Postgres or external
storage may be deferred until there is a clear operational reason to add it.

High-level sprint blocks:

- `0.5.0` = persistence substrate
- `0.5.1` = session and conversation models
- `0.5.2` = message store
- `0.5.3` = sessions API
- `0.5.4` = multi-turn chat with session id
- `0.5.5` = memory summary model
- `0.5.6` = conversation context injection
- `0.5.7` = tests, migration policy, and README update

## `0.6` Notes Search, Read-Only Filesystem, and RAG v1

Status: planned.

This milestone adds local retrieval over explicitly allowed read-only roots.
File access is read-only, allowed roots must be explicit, unrestricted
filesystem access is not allowed, and no shell execution is introduced here.

High-level sprint blocks:

- `0.6.0` = read-only filesystem policy and allowed roots
- `0.6.1` = file discovery and metadata indexing
- `0.6.2` = notes ingestion for text and Markdown files
- `0.6.3` = embedding pipeline using the configured embedding model
- `0.6.4` = local retrieval index
- `0.6.5` = NotesSearchTool
- `0.6.6` = ReadOnlyFileTool
- `0.6.7` = RAG context builder with source snippets
- `0.6.8` = agent integration for notes and file search
- `0.6.9` = tests, smoke cases, and README update

## `0.7` Calendar Integration

Status: planned.

Read-only calendar access comes first. Write operations require explicit
approval. Google Calendar integration is not required before local and fake
adapters are stable.

High-level sprint blocks:

- `0.7.0` = calendar contracts
- `0.7.1` = fake calendar adapter
- `0.7.2` = local ICS calendar adapter
- `0.7.3` = CalendarSearchTool
- `0.7.4` = calendar availability/query API
- `0.7.5` = optional write-intent model with approval gate
- `0.7.6` = Google Calendar adapter spike or deferred plan
- `0.7.7` = tests and README update

## `0.8` Python Sandbox

Status: planned.

Sandboxing is a security-sensitive milestone. It must not provide generic shell
access, host home directory access, or network access by default. Timeout,
memory, and output limits are required.

High-level sprint blocks:

- `0.8.0` = sandbox policy and threat model
- `0.8.1` = dry-run Python tool
- `0.8.2` = isolated execution backend selection
- `0.8.3` = resource limits
- `0.8.4` = temporary workspace management
- `0.8.5` = approval gate for code execution
- `0.8.6` = PythonSandboxTool integration
- `0.8.7` = adversarial tests and README warnings

## `0.9` Streaming and CLI

Status: planned.

Streaming is added after core runtime and agent behavior are stable. The CLI is
a local UX layer over existing runtime capabilities, and it must not bypass
runtime safety policies.

High-level sprint blocks:

- `0.9.0` = streaming response contracts
- `0.9.1` = provider streaming client support
- `0.9.2` = `/chat/stream` endpoint
- `0.9.3` = agent event stream
- `0.9.4` = CLI package entrypoint
- `0.9.5` = CLI chat and diagnostics
- `0.9.6` = CLI agent run
- `0.9.7` = tests, terminal UX polish, and README update

## `0.10` Evaluation, Hardening, and Portfolio Release

Status: planned.

This milestone should not introduce major new product capabilities. The goal is
quality, reproducibility, documentation, and portfolio-readiness.

High-level sprint blocks:

- `0.10.0` = evaluation dataset format
- `0.10.1` = tool-calling evaluation suite
- `0.10.2` = agent loop regression tests
- `0.10.3` = RAG retrieval quality checks
- `0.10.4` = latency and reliability reporting
- `0.10.5` = architecture documentation
- `0.10.6` = demo workflows
- `0.10.7` = portfolio release cleanup

## Deferred Capability Mapping

| Capability | Roadmap milestone |
| --- | --- |
| provider-neutral LLM runtime boundary | `0.2.7` |
| llama.cpp backend compatibility | `0.3.7` |
| agent loop | `0.4` |
| tool calling | `0.2` and `0.3` |
| memory | `0.5` |
| RAG | `0.6` |
| notes search | `0.6` |
| file access | `0.6`, read-only |
| calendar | `0.7` |
| Python sandbox | `0.8` |
| CLI | `0.9` |
| streaming | `0.9` |
| persisted multi-turn conversation | `0.5` |

## Sequencing Rationale

The roadmap starts with a local LLM runtime so later work has a real API,
settings layer, container path, and smoke test baseline. Deterministic tools
come next because tool contracts and execution semantics should be stable
before the LLM is allowed to request tools.

The provider-neutral LLM boundary comes before structured ToolCalls because the
structured-output flow must target the `LlmProvider` protocol rather than an
Ollama-specific payload shape. LLM-produced ToolCalls are added only after the
deterministic tool boundary and provider boundary exist.

llama.cpp compatibility comes after structured ToolCalls because there must be
a concrete structured-output behavior to compare across backends. It comes
before the bounded agent loop so the loop is not built on assumptions that only
hold for one backend.

The bounded agent loop follows structured ToolCalls and backend compatibility
because it needs a reliable way to alternate between model output and tool
observations. Memory, RAG, files, calendar, and sandboxing come after the loop
because they add state, external data, or security-sensitive execution
surfaces. Streaming and CLI come after the core runtime behavior is stable, and
the final milestone focuses on evaluation, hardening, documentation, and
release readiness.

## Contribution and Planning Note

Each `0.x` milestone should be implemented through small `0.x.y` feature
branches or commits according to the repository workflow. Each sprint should
remain narrow and independently reviewable. Documentation must distinguish
implemented behavior from planned behavior, and planned features must not be
described as currently available.
