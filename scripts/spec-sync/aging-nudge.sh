#!/usr/bin/env bash
# Phase 2 ⑥ — aging nudge. For the open sync PR on <sync-branch>, decide whether to post a "still
# open" reminder: fire only when it has been open at least MIN_AGE_DAYS and we have not nudged in the
# last MIN_GAP_DAYS. Dedup via a hidden marker in the PR body that records the day of the last nudge,
# so a stuck PR gets an escalating trickle (day 3, 6, 9…) rather than a daily nag.
#
# Usage: aging-nudge.sh <sync-branch>
# Writes $GITHUB_OUTPUT: nudge=<true|false>; when true also pr_url, thread_ts, age_days.
# Requires GH_TOKEN + gh, and a GNU `date` (the ubuntu runner has it).
set -euo pipefail

sync_branch="$1"
min_age="${MIN_AGE_DAYS:-3}"
min_gap="${MIN_GAP_DAYS:-3}"
here="$(cd "$(dirname "$0")" && pwd)"
out() { echo "$1" >> "$GITHUB_OUTPUT"; }

pr="$(gh pr list --state open --head "$sync_branch" --json number --jq '.[0].number')"
if [ -z "$pr" ]; then
  echo "No open PR on $sync_branch."
  out "nudge=false"; exit 0
fi

now_days=$(( $(date -u +%s) / 86400 ))
created="$(gh pr view "$pr" --json createdAt --jq .createdAt)"
created_days=$(( $(date -u -d "$created" +%s) / 86400 ))
age=$(( now_days - created_days ))
if [ "$age" -lt "$min_age" ]; then
  echo "PR #$pr is $age day(s) old (< $min_age); no nudge."
  out "nudge=false"; exit 0
fi

body="$(gh pr view "$pr" --json body -q .body)"
marker="spec-sync-nudged:"
last="$(printf '%s\n' "$body" | sed -n "s/.*<!-- ${marker} \\([0-9]*\\) -->.*/\\1/p" | head -n1)"
if [ -n "$last" ] && [ "$(( now_days - last ))" -lt "$min_gap" ]; then
  echo "PR #$pr nudged $(( now_days - last )) day(s) ago (< $min_gap); staying quiet."
  out "nudge=false"; exit 0
fi

# Record today's nudge (preserve any other markers, e.g. the thread-ts one).
cleaned="$(printf '%s\n' "$body" | grep -vF "<!-- ${marker}" || true)"
printf '%s\n<!-- %s %s -->\n' "$cleaned" "$marker" "$now_days" | gh pr edit "$pr" --body-file -

out "nudge=true"
out "pr_url=$(gh pr view "$pr" --json url --jq .url)"
out "thread_ts=$("$here/thread-ts.sh" get "$pr")"
out "age_days=$age"
