#!/usr/bin/env bash
# Case 2 of spec-sync Slack notifications: drift is detected, but a sync PR is ALREADY open.
#
# The open-PR path runs every hour, and because check-drift compares the live spec against `main`
# (not the PR branch), every one of those runs looks like "drift" until the PR merges. Notifying on
# each would spam the channel hourly. This script suppresses that noise and fires only on genuinely
# NEW drift that appeared *beyond* what the open PR already contains:
#
#   1. Compare the live spec (check-drift already wrote it into the working tree) against the PR
#      branch's committed snapshot. Equal -> the PR already covers it -> stay silent.
#   2. Otherwise there is new drift. Dedup by content hash via a sticky comment on the PR so we ping
#      exactly once per distinct new live-spec, not once per hourly tick.
#
# Usage: notify-new-drift.sh <spec-path> <sync-branch> <label>
# Writes to $GITHUB_OUTPUT: notify=<true|false>, and when notify=true also pr_url=<url> hash=<hash>.
# Requires GH_TOKEN, GITHUB_REPOSITORY, and GITHUB_OUTPUT in the environment.
set -euo pipefail

spec_path="$1"
sync_branch="$2"
label="$3"

out() { echo "$1" >> "$GITHUB_OUTPUT"; }

# The live spec check-drift just wrote into the working tree vs the PR branch's committed snapshot.
git fetch --quiet origin "$sync_branch"
branch_spec="$(git show "origin/${sync_branch}:${spec_path}" 2>/dev/null || true)"
if [ -n "$branch_spec" ] && [ "$branch_spec" = "$(cat "$spec_path")" ]; then
  echo "Open PR already covers the current live spec; no new drift. Staying quiet."
  out "notify=false"
  exit 0
fi

hash="$(sha256sum "$spec_path" | cut -c1-12)"
pr="$(gh pr list --state open --head "$sync_branch" --json number --jq '.[0].number')"
if [ -z "$pr" ]; then
  # The PR closed between the open-check and here; nothing to annotate.
  out "notify=false"
  exit 0
fi

# The sticky comment is both the user-facing signal and the dedup store: it always holds the latest
# new-drift hash, so "does it already mention this hash" answers "have we pinged for this drift yet".
marker="<!-- spec-sync-new-drift -->"
comments="$(gh api "repos/${GITHUB_REPOSITORY}/issues/${pr}/comments" --paginate)"
cid="$(printf '%s' "$comments" | jq -r "[.[] | select(.body | contains(\"$marker\"))][0].id // empty")"
prev="$(printf '%s' "$comments" | jq -r "[.[] | select(.body | contains(\"$marker\"))][0].body // empty")"

if printf '%s' "$prev" | grep -q "$hash"; then
  echo "Already notified for live-spec ${hash}; staying quiet."
  out "notify=false"
  exit 0
fi

body="${marker}
:warning: **New ${label} spec drift beyond this PR** (live-spec \`${hash}\`). This PR is a snapshot of earlier drift and is now stale relative to the live spec — merge or close it and the next spec-sync run will open a fresh PR covering the rest."

if [ -n "$cid" ]; then
  gh api -X PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${cid}" -f body="$body" >/dev/null
else
  gh api -X POST "repos/${GITHUB_REPOSITORY}/issues/${pr}/comments" -f body="$body" >/dev/null
fi

out "notify=true"
out "pr_url=$(gh pr view "$pr" --json url --jq .url)"
out "hash=${hash}"
