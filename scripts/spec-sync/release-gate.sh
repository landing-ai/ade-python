#!/usr/bin/env bash
# Block a release while the SDK implements a route that is not yet in PRODUCTION.
# (staging-only features hold the release train until they reach prod or are reverted.)
set -euo pipefail

prod_url="https://api.va.landing.ai/v1/ade/openapi.json"
here="$(cd "$(dirname "$0")" && pwd)"
prod="$(mktemp)"
trap 'rm -f "$prod"' EXIT
"$here/fetch-normalize.sh" "$prod_url" > "$prod"

# The committed staging snapshot is the surface we ship code for; every route in it
# must also exist in production before we release. A route is a (path, method) pair,
# so we compare at method granularity — a path present in prod but missing a specific
# method (e.g. a newly added POST) must still block the release.
missing="$(jq -r --slurpfile p "$prod" '
  ($p[0].paths) as $prod
  | .paths
  | to_entries[]
  | .key as $path
  | (.value | keys[]) as $method
  | select(["get","put","post","delete","patch","options","head","trace"] | index($method))
  | select(($prod[$path] // {} | has($method)) | not)
  | "\($method | ascii_upcase) \($path)"
' specs/v1-ade.json)"

if [ -n "$missing" ]; then
  echo "release-gate: these implemented routes are not in production yet:" >&2
  echo "$missing" >&2
  exit 1
fi
echo "release-gate: all implemented routes exist in production."
