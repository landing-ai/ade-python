#!/usr/bin/env bash
# Fetch a live OpenAPI spec and emit normalized (stable, key-sorted) JSON to stdout.
# Normalization makes byte-diffs meaningful: same spec content -> identical bytes.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: fetch-normalize.sh <spec-url>" >&2
  exit 2
fi

url="$1"
curl -fsSL --max-time 30 --retry 3 --retry-delay 2 "$url" | jq -S .
