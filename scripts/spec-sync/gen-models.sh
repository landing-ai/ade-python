#!/usr/bin/env bash
# Generate REFERENCE pydantic models from a committed spec snapshot.
# Output is committed under specs/_generated/ as an input for the AI wiring phase
# and for human review — it is NOT shipped and does not replace src/landingai_ade/types/*.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: gen-models.sh <spec-path> <out-path>" >&2
  exit 2
fi

spec="$1"
out="$2"
mkdir -p "$(dirname "$out")"
# --openapi-scopes: the default is `schemas` only, which emits models for
# components/schemas but SKIPS request/response bodies defined inline in a path
# (this spec inlines them — e.g. the build-schema request body is not a named
# component). Without `paths` the entire REQUEST-INPUT surface is invisible to the
# AI wiring phase, so it never sees multipart/file-upload variants or input
# required/optional. Adding `paths parameters` emits per-operation *PostRequest /
# *PostResponse / *Query models covering that surface.
rye run datamodel-codegen \
  --input "$spec" \
  --input-file-type openapi \
  --output "$out" \
  --output-model-type pydantic_v2.BaseModel \
  --openapi-scopes schemas paths parameters \
  --use-standard-collections \
  --use-schema-description \
  --field-constraints \
  --disable-timestamp
