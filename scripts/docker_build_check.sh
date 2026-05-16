#!/bin/sh
set -eu

IMAGE_NAME="${IMAGE_NAME:-aigentego:docker-build-check}"
CONTAINER_NAME="${CONTAINER_NAME:-aigentego-build-check-$$}"

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

docker build -t "$IMAGE_NAME" .
cleanup

docker run \
    -d \
    --name "$CONTAINER_NAME" \
    -e AGENT_HOST=0.0.0.0 \
    -e AGENT_PORT=8080 \
    -e OLLAMA_BASE_URL=http://127.0.0.1:1 \
    "$IMAGE_NAME" >/dev/null

attempts=30
while [ "$attempts" -gt 0 ]; do
    if docker exec "$CONTAINER_NAME" python -c 'import json, urllib.request; data = json.load(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)); assert data["status"] == "ok"' 2>/dev/null; then
        exit 0
    fi

    state="$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    if [ "$state" = "exited" ]; then
        docker logs "$CONTAINER_NAME"
        exit 1
    fi

    attempts=$((attempts - 1))
    sleep 1
done

docker logs "$CONTAINER_NAME"
exit 1
