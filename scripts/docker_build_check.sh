#!/bin/sh
set -eu

IMAGE_NAME="${IMAGE_NAME:-aigentego:docker-build-check}"

docker build -t "$IMAGE_NAME" .
docker run --rm "$IMAGE_NAME"
