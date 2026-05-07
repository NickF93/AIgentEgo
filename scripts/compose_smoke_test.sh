#!/bin/sh
set -eu

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-aigentego-smoke}"
CHAT_MODEL="${OLLAMA_CHAT_MODEL:-llama3.2:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

export OLLAMA_HOST_PORT="${OLLAMA_HOST_PORT:-0}"

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

trap cleanup EXIT INT TERM

cleanup

docker compose -p "$PROJECT_NAME" up --build -d ollama ollama-init
wait_for_healthy ollama
wait_for_successful_exit ollama-init

wait_for_model "$CHAT_MODEL"
wait_for_model "$EMBED_MODEL"

docker compose -p "$PROJECT_NAME" up --build --exit-code-from agent-api agent-api
