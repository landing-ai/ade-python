#!/usr/bin/env bash
# Cross-check the V2 resource sources against the committed V2 spec snapshot, in BOTH directions:
#
#   1. Every URL path a V2 resource sends (a quoted `/v2/...` or `/v1/...` literal) must exist under
#      `paths` in the snapshot. This catches a wired route the spec does not have. It is the exact
#      failure of ade-python#153: the spec added /v1/classify and /v1/split (V1-compatibility routes
#      of the AIDE gateway), and the AI-wiring pass "translated" them into client.v2.classify /
#      client.v2.split hitting /v2/classify and /v2/split — paths that exist nowhere. Staging 404'd
#      and the contract gate went red, but nothing said WHY until a human read the spec.
#   2. Every `/v2/*` path in the snapshot must be sent by some V2 resource. This catches a new /v2
#      route the wiring pass skipped. A route that is deliberately not wired goes in DEFERRED below,
#      or is stripped from the snapshot in fetch-normalize.sh (how build-schema is hidden).
#
# Path parameters compare structurally: `{job_id}` in the spec, `{job_id}` in a Python f-string and
# `${jobID}` in a TypeScript template literal all normalize to `{}`.
#
# Runs in `./scripts/lint` (so the CI `lint` job enforces it on every PR) and inside the spec-sync
# workflow's lint step, where a failure is fed back to the AI repair pass. It only READS source:
# grep + jq over files, nothing is imported or executed — so it is safe to run over AI-authored code
# in a trusted step, like ruff/pyright.
#
# usage: check-v2-paths.sh <spec-snapshot.json> <v2-resources-dir>
#   exit 0 -> consistent
#   exit 1 -> mismatch (details on stdout, one line per finding)
#   exit 2 -> usage error
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: check-v2-paths.sh <spec-snapshot.json> <v2-resources-dir>" >&2
  exit 2
fi

spec="$1"
srcdir="$2"
[ -f "$spec" ] || { echo "check-v2-paths: spec snapshot not found: $spec" >&2; exit 2; }
[ -d "$srcdir" ] || { echo "check-v2-paths: resources dir not found: $srcdir" >&2; exit 2; }

# Wired on purpose but absent from the snapshot: the hidden build-schema surface. Keep in sync with
# the `del(...)` list in fetch-normalize.sh — the resource module is retained (not public), so its
# URL literals stay in src/ while the routes are stripped from specs/v2-aide.json.
HIDDEN='/v2/extract/build-schema
/v2/extract/build-schema/jobs
/v2/extract/build-schema/jobs/{}'

# In the snapshot on purpose but NOT wired: deferred surface a maintainer has decided to skip for
# now. /v2/workflow* is kept in the baseline as a drift signal but intentionally not implemented in
# this SDK (the AI-wiring prompt excludes it too). Add a path here (normalized: parameters as `{}`)
# only as an explicit product decision; remove it when the surface ships.
DEFERRED='/v2/workflow
/v2/workflow/jobs
/v2/workflow/jobs/{}'

normalize() {
  # `${jobID}` (TS template) and `{job_id}` (spec / Python f-string) -> `{}`
  sed -E 's/\$\{[^}]*\}/{}/g; s/\{[^}]*\}/{}/g'
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

jq -r '.paths | keys[]' "$spec" | normalize | sort -u > "$tmp/spec"
printf '%s\n' "$HIDDEN" | sed '/^$/d' | sort -u > "$tmp/hidden"
printf '%s\n' "$DEFERRED" | sed '/^$/d' | sort -u > "$tmp/deferred"

# Every quoted `/vN/...` literal in the resource sources, as `<file>:<line>\t<normalized path>`.
# Comment lines (`#`, `//`, `*`) are skipped so prose that mentions a path is not mistaken for a
# request. Quote characters: " ' and ` (TS template literals).
quote="[\"'\`]"
literal="${quote}/v[0-9]+/[^\"'\`[:space:]]*${quote}"
: > "$tmp/wired_loc"
grep -rnE --include='*.py' --include='*.ts' --exclude='*.test.ts' --exclude='*.d.ts' \
  -e "$literal" "$srcdir" 2>/dev/null \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(#|//|\*)' \
  | sed -E $'s/^([^:]+:[0-9]+):(.*)$/\\1\t\\2/' \
  | while IFS=$'\t' read -r loc text; do
      printf '%s\n' "$text" | grep -oE "$literal" | sed -E 's/^.//; s/.$//' | normalize \
        | while IFS= read -r p; do printf '%s\t%s\n' "$loc" "$p"; done
    done > "$tmp/wired_loc" || true
cut -f2 "$tmp/wired_loc" | sort -u > "$tmp/wired"

status=0

# 1. wired ⊆ spec ∪ hidden
sort -u "$tmp/spec" "$tmp/hidden" > "$tmp/known"
comm -23 "$tmp/wired" "$tmp/known" > "$tmp/unknown"
if [ -s "$tmp/unknown" ]; then
  status=1
  while IFS= read -r p; do
    echo "check-v2-paths: ERROR — a V2 resource sends '$p', but '$spec' has no such path under 'paths'."
    echo "  A client.v2 method must call a route that exists in the spec. Never invent or translate a"
    echo "  path (e.g. a /v1/... route rewritten as /v2/...): if the spec has no /v2 route for this"
    echo "  capability it is out of scope for client.v2 — remove the method/resource, its registration,"
    echo "  types, tests and docs. Sent from:"
    awk -F'\t' -v p="$p" '$2 == p { print "    " $1 }' "$tmp/wired_loc"
  done < "$tmp/unknown"
fi

# 2. spec[/v2/*] ⊆ wired ∪ deferred
grep '^/v2/' "$tmp/spec" > "$tmp/spec_v2" || true
sort -u "$tmp/wired" "$tmp/deferred" > "$tmp/covered"
comm -23 "$tmp/spec_v2" "$tmp/covered" > "$tmp/unwired"
if [ -s "$tmp/unwired" ]; then
  status=1
  while IFS= read -r p; do
    echo "check-v2-paths: ERROR — '$spec' has the route '$p' under 'paths', but no V2 resource in '$srcdir' sends it."
    echo "  Wire it (mirror the parse/extract resources), or record an explicit decision to skip it"
    echo "  in DEFERRED in scripts/spec-sync/check-v2-paths.sh."
  done < "$tmp/unwired"
fi

if [ "$status" -eq 0 ]; then
  echo "check-v2-paths: OK — $(wc -l < "$tmp/wired" | tr -d ' ') wired path(s) all present in $spec; all $(wc -l < "$tmp/spec_v2" | tr -d ' ') /v2 spec route(s) wired."
fi
exit "$status"
