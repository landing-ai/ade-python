#!/usr/bin/env bash
# Fetch a live OpenAPI spec and emit normalized (stable, key-sorted) JSON to stdout.
# Normalization makes byte-diffs meaningful: same spec content -> identical bytes.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: fetch-normalize.sh <spec-url>" >&2
  exit 2
fi

url="$1"
body="$(mktemp)"
trap 'rm -f "$body"' EXIT
# Fetch WITHOUT `-f` so we can inspect the HTTP status ourselves and separate "expected on staging"
# from "real problem" — `-f` collapses every 4xx/5xx into one exit code. A transport failure (DNS,
# connection refused, timeout — curl's own non-zero exit) means the source is unreachable, which the
# caller treats as an expected no-op: exit 20.
code="$(curl -sS --max-time 30 --retry 3 --retry-delay 2 -o "$body" -w '%{http_code}' "$url")" || exit 20
# Map the HTTP status:
#   200  -> normalize below.
#   404  -> an unbooked/torn-down staging cluster serves 404 (the spec route disappears with the
#           backend); this is the EXPECTED no-op, so exit 20 as well.
#   else -> a reachable source returning 401/403 (auth), 5xx (server error), etc. is a REAL problem;
#           exit 1 (operational error) so the workflow's catch-all alert fires.
case "$code" in
  200) : ;;
  404) exit 20 ;;
  *) echo "fetch-normalize: unexpected HTTP $code from $url" >&2; exit 1 ;;
esac
# `jq -S` sorts object keys only; array element order (e.g. `required`, `enum`, `tags`) is
# preserved as emitted by the backend. This assumes the gateway emits arrays deterministically.
# If it ever reorders them, drift detection would fire on cosmetic churn (phantom PRs) — start
# debugging false drift here.
#
# `del(...)`: the V2 build-schema endpoints are intentionally hidden from the SDK
# (see the client-surface PR). Stripping them during normalization keeps them out of the
# committed snapshot, so spec-sync's AI wiring never sees them and cannot re-add the hidden
# surface. To un-hide, remove this filter and re-wire the resource. NOTE: V1
# `/v1/ade/extract/build-schema` is deliberately NOT stripped — only the V2 surface is hidden.
#
# A 200 with an empty/whitespace body would make `jq` emit nothing and exit 0 — a silent "empty spec"
# that check-drift would treat as drift and commit, clobbering the snapshot. Reject it loudly instead.
# (Malformed JSON already fails via jq's non-zero exit.)
normalized="$(jq -S '
  del(
    .paths["/v2/extract/build-schema"],
    .paths["/v2/extract/build-schema/jobs"],
    .paths["/v2/extract/build-schema/jobs/{job_id}"]
  )' < "$body")"
if [ -z "$normalized" ]; then
  echo "fetch-normalize: empty/blank spec from $url" >&2
  exit 1
fi
printf '%s\n' "$normalized"
