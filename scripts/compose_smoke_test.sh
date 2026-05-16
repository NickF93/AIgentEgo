#!/bin/sh
set -eu

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-aigentego-smoke}"
CHAT_MODEL="${OLLAMA_CHAT_MODEL:-llama3.2:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
AGENT_PORT="${AGENT_PORT:-8080}"

export OLLAMA_HOST_PORT="${OLLAMA_HOST_PORT:-0}"
export AGENT_HOST_PORT="${AGENT_HOST_PORT:-0}"
export AGENT_PORT

cleanup() {
    docker compose -p "$PROJECT_NAME" down --remove-orphans >/dev/null 2>&1 || true
}

wait_for_healthy() {
    service="$1"
    attempts="${2:-90}"

    while [ "$attempts" -gt 0 ]; do
        container_id="$(docker compose -p "$PROJECT_NAME" ps -q "$service" 2>/dev/null || true)"

        if [ -n "$container_id" ]; then
            status="$(
                docker inspect \
                    --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
                    "$container_id" 2>/dev/null || true
            )"

            if [ "$status" = "healthy" ]; then
                return 0
            fi
        fi

        attempts=$((attempts - 1))
        sleep 2
    done

    docker compose -p "$PROJECT_NAME" logs "$service"
    return 1
}

wait_for_successful_exit() {
    service="$1"
    attempts="${2:-300}"

    while [ "$attempts" -gt 0 ]; do
        container_id="$(docker compose -p "$PROJECT_NAME" ps -a -q "$service" 2>/dev/null || true)"

        if [ -n "$container_id" ]; then
            state="$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null || true)"
            exit_code="$(docker inspect --format '{{.State.ExitCode}}' "$container_id" 2>/dev/null || true)"

            if [ "$state" = "exited" ]; then
                if [ "$exit_code" = "0" ]; then
                    return 0
                fi

                docker compose -p "$PROJECT_NAME" logs "$service"
                return 1
            fi
        fi

        attempts=$((attempts - 1))
        sleep 2
    done

    docker compose -p "$PROJECT_NAME" logs "$service"
    return 1
}

model_available() {
    model="$1"
    models="$(docker compose -p "$PROJECT_NAME" exec -T ollama ollama list | awk 'NR > 1 {print $1}')"

    if printf '%s\n' "$models" | grep -Fx "$model" >/dev/null; then
        return 0
    fi

    case "$model" in
        *:*)
            return 1
            ;;
        *)
            printf '%s\n' "$models" | grep -Fx "$model:latest" >/dev/null
            ;;
    esac
}

wait_for_model() {
    model="$1"
    attempts="${2:-30}"

    while [ "$attempts" -gt 0 ]; do
        if model_available "$model"; then
            return 0
        fi

        attempts=$((attempts - 1))
        sleep 2
    done

    printf 'Configured model is not available: %s\n' "$model" >&2
    docker compose -p "$PROJECT_NAME" exec -T ollama ollama list >&2
    return 1
}

agent_get() {
    path="$1"
    docker compose -p "$PROJECT_NAME" exec -T agent-api python -c '
import json
import sys
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:" + sys.argv[2] + sys.argv[1],
    timeout=10,
) as response:
    data = json.load(response)

if not isinstance(data, dict) or data.get("status") != "ok":
    raise SystemExit("unexpected response: " + repr(data))
' "$path" "$AGENT_PORT"
}

wait_for_agent_api() {
    attempts="${1:-60}"

    while [ "$attempts" -gt 0 ]; do
        if agent_get /health >/dev/null 2>&1; then
            return 0
        fi

        container_id="$(docker compose -p "$PROJECT_NAME" ps -q agent-api 2>/dev/null || true)"
        if [ -n "$container_id" ]; then
            state="$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null || true)"
            if [ "$state" = "exited" ]; then
                docker compose -p "$PROJECT_NAME" logs agent-api
                return 1
            fi
        fi

        attempts=$((attempts - 1))
        sleep 2
    done

    docker compose -p "$PROJECT_NAME" logs agent-api
    return 1
}

agent_chat() {
    docker compose -p "$PROJECT_NAME" exec -T agent-api python -c '
import json
import sys
import urllib.request

payload = json.dumps({"message": "Reply with one short sentence."}).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:" + sys.argv[1] + "/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=180) as response:
    data = json.load(response)

if not isinstance(data, dict):
    raise SystemExit("chat response must be a JSON object")
if not data.get("request_id"):
    raise SystemExit("chat response is missing request_id")
if not data.get("model"):
    raise SystemExit("chat response is missing model")
if not str(data.get("message", "")).strip():
    raise SystemExit("chat response message is empty")
' "$AGENT_PORT"
}

trap cleanup EXIT INT TERM

cleanup

docker compose -p "$PROJECT_NAME" up --build -d ollama ollama-init
wait_for_healthy ollama
wait_for_successful_exit ollama-init

wait_for_model "$CHAT_MODEL"
wait_for_model "$EMBED_MODEL"

docker compose -p "$PROJECT_NAME" up --build -d agent-api
wait_for_agent_api
agent_get /health
agent_get /diagnostics
agent_chat
