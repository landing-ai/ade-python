#!/usr/bin/env bash
# Fetch a live OpenAPI spec and emit normalized (stable, key-sorted) JSON to stdout.
# Normalization makes byte-diffs meaningful: same spec content -> identical bytes.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: fetch-normalize.sh <spec-url>" >&2
  exit 2
fi

url="$1"
# Fetch first, separately from normalization, so the two failure modes stay distinguishable. A fetch
# failure (source unreachable — e.g. an unbooked staging cluster) exits 20, which the caller treats
# as an expected no-op; a reachable-but-invalid spec still fails loudly below.
raw="$(curl -fsSL --max-time 30 --retry 3 --retry-delay 2 "$url")" || exit 20
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
# (Malformed JSON already fails via jq's non-zero exit under pipefail.)
normalized="$(printf '%s' "$raw" | jq -S '
  del(
    .paths["/v2/extract/build-schema"],
    .paths["/v2/extract/build-schema/jobs"],
    .paths["/v2/extract/build-schema/jobs/{job_id}"]
  )')"
if [ -z "$normalized" ]; then
  echo "fetch-normalize: empty/blank spec from $url" >&2
  exit 1
fi
printf '%s\n' "$normalized"
