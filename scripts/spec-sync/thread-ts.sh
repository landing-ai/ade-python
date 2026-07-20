#!/usr/bin/env bash
# Persist / retrieve the Slack thread-root `ts` for a sync PR, stored as a hidden marker in the PR
# body. Lifecycle events in later runs and other workflows (new drift, gates red, merged) read it
# back to reply into the same thread instead of posting a fresh top-level message.
#
#   thread-ts.sh get <pr>       -> print the stored ts (empty if none)
#   thread-ts.sh set <pr> <ts>  -> upsert the marker in the PR body
#
# Only the run that opens the PR writes; every later event only reads, so there is no write race
# (and spec-sync is serialized by its concurrency group anyway). Requires GH_TOKEN + gh.
set -euo pipefail

marker="spec-sync-slack-thread:"
cmd="${1:?usage: thread-ts.sh get|set <pr> [ts]}"
pr="${2:?pr number required}"

case "$cmd" in
  get)
    # Best-effort: a read failure just means "no thread root yet" -> caller posts top-level.
    body="$(gh pr view "$pr" --json body -q .body 2>/dev/null || true)"
    printf '%s\n' "$body" | sed -n "s/.*<!-- ${marker} \\([^ ]*\\) -->.*/\\1/p" | head -n1
    ;;
  set)
    ts="${3:?ts required}"
    # Fail closed: a `gh pr view` failure must NOT look like an empty body, or the `gh pr edit`
    # below would replace the entire PR description with just the marker. Abort instead. (An
    # actually-empty description is a success with empty output, and is handled fine.)
    if ! body="$(gh pr view "$pr" --json body -q .body)"; then
      echo "thread-ts.sh: could not read PR #$pr body; refusing to overwrite it." >&2
      exit 1
    fi
    # Drop any existing marker line, then append the fresh one, and replace the PR body.
    cleaned="$(printf '%s\n' "$body" | grep -vF "<!-- ${marker}" || true)"
    printf '%s\n<!-- %s %s -->\n' "$cleaned" "$marker" "$ts" | gh pr edit "$pr" --body-file -
    ;;
  *)
    echo "thread-ts.sh: unknown command '$cmd' (want get|set)" >&2
    exit 2
    ;;
esac
