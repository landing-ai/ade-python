#!/usr/bin/env bash
# Cross-check the V2 resource sources against the committed V2 spec snapshot, in BOTH directions:
#
#   1. FORWARD — every URL path a V2 resource mentions in code (a quoted `/v2/...` or `/v1/...`
#      literal outside comment lines) must be a `/v2/*` path under `paths` in the snapshot (or a
#      HIDDEN path, see below). Two failure classes, reported separately:
#        - not in the spec at all: a wired route the spec does not have. This is the exact failure
#          of landing-ai/ade-python#153: the spec added /v1/classify and /v1/split (V1-compat routes
#          of the AIDE gateway) and the AI-wiring pass "translated" them into client.v2.classify /
#          client.v2.split hitting /v2/classify and /v2/split — paths that exist nowhere.
#        - in the spec but not `/v2/*`: a V1-compatibility route wired into client.v2 directly.
#          Only `/v2/*` backs client.v2; the `/v1/*` routes in this spec are the V1 surface.
#      The forward scan is deliberately broad (any quoted literal, comments excluded) so a hardcoded
#      URL that bypasses the builder is still caught; prose that quotes a non-existent path fails
#      too, which is acceptable — it is misleading prose.
#   2. REVERSE — every `/v2/*` path in the snapshot must be SENT by some V2 resource. "Sent" means
#      the literal sits at a request call site: the V2 URL builder (`_v2_url(` in Python, `v2Url(`
#      in TypeScript) as the first argument of a request method (`self._get/_post/...(` in Python,
#      `this._client.get/post/...(` in TypeScript). A route that is only mentioned in a docstring or
#      comment, or only passed through the builder without being sent, does not count. The scan is
#      whitespace-insensitive across lines, so a formatter splitting the call is fine. Catches a new
#      /v2 route the wiring pass skipped. Namespaces the AI pass is told not to wire
#      (NOT_AUTO_WIRED_PREFIXES) are excluded from this direction only; routes hidden from the
#      snapshot in fetch-normalize.sh (build-schema) never appear in it.
#
# Path parameters compare structurally: `{job_id}` in the spec, `{job_id}` in a Python f-string and
# `${jobID}` in a TypeScript template literal all normalize to `{}`.
#
# Runs in `./scripts/lint` (so the CI `lint` job enforces it on every PR) and inside the spec-sync
# workflow's lint step, where a failure is fed back to the AI repair pass. It only READS source:
# grep + a JSON key dump over files, nothing is imported or executed — so it is safe to run over
# AI-authored code in a trusted step, like the linters. Reading the snapshot's `paths` keys uses
# whichever of jq, python3 or node is on PATH, so a plain checkout needs no extra tool: the Python
# repo always has python3 and the TypeScript repo always has node. This file is identical in
# ade-python and ade-typescript.
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

# Snapshot namespaces the AI-wiring prompt explicitly does NOT wire (hand-maintained in
# ade-typescript, deferred in ade-python). Excluded from the REVERSE check only, so a new route
# under them does not fail lint and push the repair pass into wiring what the prompt forbids. A
# path under them that IS wired is still validated by the forward check. One prefix per line.
NOT_AUTO_WIRED_PREFIXES='/v2/workflow'

# Print the snapshot's `paths` keys, one per line, with whatever JSON reader is available.
spec_paths() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '.paths | keys[]' "$1"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json, sys
for key in json.load(open(sys.argv[1], encoding="utf-8"))["paths"]:
    print(key)' "$1"
  elif command -v node >/dev/null 2>&1; then
    node -e 'for (const k of Object.keys(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).paths)) console.log(k)' "$1"
  else
    echo "check-v2-paths: need jq, python3 or node on PATH to read $1" >&2
    exit 2
  fi
}

normalize() {
  # `${jobID}` (TS template) and `{job_id}` (spec / Python f-string) -> `{}`
  sed -E 's/\$\{[^}]*\}/{}/g; s/\{[^}]*\}/{}/g'
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

spec_paths "$spec" | normalize | sort -u > "$tmp/spec"
grep '^/v2/' "$tmp/spec" > "$tmp/spec_v2" || true
printf '%s\n' "$HIDDEN" | sed '/^$/d' | sort -u > "$tmp/hidden"

# Lines that are comments: `#`, `//`, a block-comment opener `/*` / `/**`, or a `*` continuation.
comment_line_re='^[[:space:]]*(#|//|/\*|\*)'
quote="[\"'\`]"
literal="${quote}/v[0-9]+/[^\"'\`[:space:]]*${quote}"
# A request call site, matched against a file whose comment lines were dropped and whose remaining
# lines were joined with spaces (so the pattern spans formatter line breaks):
#   Python:      self._post(  self._v2_url("/v2/parse")      self._get(  self._v2_url(f"/v2/x/{id}")
#   TypeScript:  this._client.post<T>(  this.v2Url('/v2/parse')   this._client.get<T>(this.v2Url(path`/v2/x/${id}`)
methods='(get|post|put|patch|delete)'
callsite="(\\._${methods}|\\._client\\.${methods}(<[^(]*>)?)\\([[:space:]]*(self|this)\\.(_v2_url|v2Url)\\([[:space:]]*(path)?[[:space:]]*f?${literal}"

find "$srcdir" -type f \( -name '*.py' -o -name '*.ts' \) ! -name '*.test.ts' ! -name '*.d.ts' | sort > "$tmp/files"

# FORWARD input: every literal on a non-comment line, as `<file>:<line>\t<normalized path>`.
: > "$tmp/mentioned_loc"
while IFS= read -r file; do
  # `|| true`: a file without any literal makes grep exit 1, which under pipefail must not abort.
  grep -nE -e "$literal" "$file" 2>/dev/null \
    | grep -vE "^[0-9]+:${comment_line_re#^}" \
    | while IFS= read -r hit; do
        lineno="${hit%%:*}"
        text="${hit#*:}"
        printf '%s\n' "$text" | grep -oE "$literal" | sed -E 's/^.//; s/.$//' | normalize \
          | while IFS= read -r p; do printf '%s:%s\t%s\n' "$file" "$lineno" "$p"; done
      done || true
done < "$tmp/files" >> "$tmp/mentioned_loc"
cut -f2 "$tmp/mentioned_loc" | sort -u > "$tmp/mentioned"

# REVERSE input: literals at a request call site, scanned over the comment-stripped, joined file.
: > "$tmp/sent"
while IFS= read -r file; do
  grep -vE "$comment_line_re" "$file" | tr '\n' ' ' | grep -oE "$callsite" | grep -oE "$literal" \
    | sed -E 's/^.//; s/.$//' | normalize || true
done < "$tmp/files" | sort -u > "$tmp/sent"

status=0

# 1. FORWARD: mentioned ⊆ spec[/v2/*] ∪ hidden
sort -u "$tmp/spec_v2" "$tmp/hidden" > "$tmp/known"
comm -23 "$tmp/mentioned" "$tmp/known" > "$tmp/unknown"
if [ -s "$tmp/unknown" ]; then
  status=1
  while IFS= read -r p; do
    if grep -qxF "$p" "$tmp/spec"; then
      echo "check-v2-paths: ERROR — a V2 resource uses '$p', which is in '$spec' but is NOT a /v2 route: it is out of scope for client.v2."
      echo "  Only the spec's /v2/* routes back client.v2. The /v1/* routes in this spec are the V1-compatibility"
      echo "  surface (tracked by the V1 spec-sync job) and must not be wired into client.v2 — remove the"
      echo "  method/resource, its registration, types, tests and docs. Used at:"
    else
      echo "check-v2-paths: ERROR — a V2 resource uses '$p', but '$spec' has no such path under 'paths'."
      echo "  A client.v2 method must call a route that exists in the spec. Never invent or translate a"
      echo "  path (e.g. a /v1/... route rewritten as /v2/...): if the spec has no /v2 route for this"
      echo "  capability it is out of scope for client.v2 — remove the method/resource, its registration,"
      echo "  types, tests and docs. Used at:"
    fi
    awk -F'\t' -v p="$p" '$2 == p { print "    " $1 }' "$tmp/mentioned_loc"
  done < "$tmp/unknown"
fi

# 2. REVERSE: spec[/v2/*] minus not-auto-wired namespaces ⊆ sent
cp "$tmp/spec_v2" "$tmp/must_wire"
while IFS= read -r prefix; do
  [ -n "$prefix" ] || continue
  grep -vE "^$(printf '%s' "$prefix" | sed -E 's/[][\.*^$/]/\\&/g')(/|$)" "$tmp/must_wire" > "$tmp/must_wire.next" || true
  mv "$tmp/must_wire.next" "$tmp/must_wire"
done <<< "$NOT_AUTO_WIRED_PREFIXES"
comm -23 "$tmp/must_wire" "$tmp/sent" > "$tmp/unwired"
if [ -s "$tmp/unwired" ]; then
  status=1
  while IFS= read -r p; do
    echo "check-v2-paths: ERROR — '$spec' has the route '$p' under 'paths', but no V2 resource in '$srcdir' sends it."
    echo "  \"Sends\" means a request call site — the V2 URL builder as the first argument of a request method"
    echo "  (self._post(self._v2_url(...)) / this._client.post(this.v2Url(...))) — not a mention in a comment"
    echo "  or docstring. Wire it (mirror the parse/extract resources), or — as an explicit product decision —"
    echo "  add its namespace to NOT_AUTO_WIRED_PREFIXES in scripts/spec-sync/check-v2-paths.sh."
  done < "$tmp/unwired"
fi

if [ "$status" -eq 0 ]; then
  echo "check-v2-paths: OK — $(wc -l < "$tmp/mentioned" | tr -d ' ') path(s) used by V2 resources are all /v2 routes of $spec (or hidden); all $(wc -l < "$tmp/must_wire" | tr -d ' ') auto-wired /v2 spec route(s) are sent."
fi
exit "$status"
