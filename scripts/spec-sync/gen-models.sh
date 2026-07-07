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
rye run datamodel-codegen \
  --input "$spec" \
  --input-file-type openapi \
  --output "$out" \
  --output-model-type pydantic_v2.BaseModel \
  --use-standard-collections \
  --use-schema-description \
  --field-constraints \
  --disable-timestamp
