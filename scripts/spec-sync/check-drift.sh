#!/usr/bin/env bash
# Compare a live spec against its committed snapshot.
#   exit 0  -> no drift
#   exit 10 -> drift detected; <committed-path> updated in place with the live spec
#   other   -> operational error (e.g. fetch failure)
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: check-drift.sh <spec-url> <committed-path>" >&2
  exit 2
fi

url="$1"
committed="$2"
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

"$here/fetch-normalize.sh" "$url" > "$tmp"

if [ -f "$committed" ] && diff -q "$committed" "$tmp" >/dev/null; then
  echo "no drift: $committed"
  exit 0
fi

mkdir -p "$(dirname "$committed")"
cp "$tmp" "$committed"
echo "drift detected -> updated $committed"
exit 10
