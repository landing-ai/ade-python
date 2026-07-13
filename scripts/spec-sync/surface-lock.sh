#!/usr/bin/env bash
# Fail if the RELEASED public API of landingai_ade changed in a breaking way.
# Baseline = last release tag: released surface is the promise to users; merged-but-
# unreleased surface stays mutable.
#
# Exemption: a change to the *value* of a public constant (e.g. adding an entry to the
# ENVIRONMENTS / V2_ENVIRONMENTS maps) is additive configuration, not an API-signature
# break, so griffe's "Attribute value was changed" breakages are treated as non-breaking.
# A genuine intentional break can still be approved via the `breaking-change-approved`
# PR label (see .github/workflows/pr-gates.yml).
set -uo pipefail

baseline="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
if [ -z "$baseline" ]; then
  echo "surface-lock: no release tag found; skipping (nothing released yet)."
  exit 0
fi

echo "surface-lock: checking landingai_ade against $baseline"
report="$(rye run griffe check landingai_ade --against "$baseline" --search src -f oneline 2>&1)"
status=$?

if [ "$status" -eq 0 ]; then
  echo "surface-lock: no breaking changes."
  exit 0
fi

# griffe found breakages (one per line). Drop attribute-value-only changes (additive
# public constants); anything left is a real signature/API break.
breaking="$(printf '%s\n' "$report" | grep -vi 'Attribute value was changed' | grep -vE '^[[:space:]]*$' || true)"

if [ -z "$breaking" ]; then
  echo "surface-lock: only public-constant value changes (additive); treating as non-breaking:"
  printf '%s\n' "$report" | sed 's/^/  (ignored) /'
  exit 0
fi

echo "surface-lock: BREAKING changes to the released public API:"
printf '%s\n' "$breaking"
exit 1
