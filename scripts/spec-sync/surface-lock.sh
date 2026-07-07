#!/usr/bin/env bash
# Fail if the RELEASED public API of landingai_ade changed in a breaking way.
# Baseline = last release tag: released surface is the promise to users; merged-but-
# unreleased surface stays mutable.
set -euo pipefail

baseline="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
if [ -z "$baseline" ]; then
  echo "surface-lock: no release tag found; skipping (nothing released yet)."
  exit 0
fi

echo "surface-lock: checking landingai_ade against $baseline"
rye run griffe check landingai_ade --against "$baseline" --search src
