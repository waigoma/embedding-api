#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   FLAVOR=nvidia ./build.sh
#   FLAVOR=amd ./build.sh
FLAVOR="${FLAVOR:-nvidia}"

case "${FLAVOR}" in
  nvidia|amd) ;;
  *)
    echo "Unsupported FLAVOR: ${FLAVOR} (use: nvidia|amd)"
    exit 1
    ;;
esac

docker compose -f compose.yml -f "compose.build.${FLAVOR}.yml" build
