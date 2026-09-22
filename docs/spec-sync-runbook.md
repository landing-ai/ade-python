# spec-sync runbook

Operational guide for whoever is on point for this SDK: what to watch, what to decide, and who picks
it up after you merge.

The **mechanism** — how the pipeline is built, what each gate checks, why the token cannot be
`GITHUB_TOKEN` — lives in [CONTRIBUTING.md](../CONTRIBUTING.md#spec-sync-pipeline). This file is
only the part a human has to do. `ade-typescript` carries the mirror of this runbook; the two SDKs
are maintained as a pair and most decisions here apply to both.

## The loop in one paragraph

`Spec Sync` runs hourly (`.github/workflows/spec-sync.yml`, cron `0 * * * *`). When the live
**staging** spec drifts from the committed snapshot, it opens one PR on a fixed branch
(`spec-sync/v1` or `spec-sync/v2`) and announces it in Slack. You review and merge it — the wiring
commit is AI-drafted and **always** needs human review. Merging publishes nothing: QA tests merged
`main` against staging first, and a maintainer dispatches the release only after the API has reached
production.

## 1. Where the signal is: `#ade-sdk-pipeline`

Every spec-sync message goes to **`#ade-sdk-pipeline`** (the default channel in
`.github/actions/slack-notify/action.yml`), and messages are **threaded**: the "PR opened" message
is the thread root and everything afterwards is a reply under it. The root's `ts` is persisted in
the PR body by `scripts/spec-sync/thread-ts.sh` so later runs and the lifecycle workflow can find
the thread.

**The channel looks quieter than it is — open the threads.** Only the "PR merged" reply broadcasts
back to the channel.

V1 and V2 are independent loops with their own PRs and their own threads. Every message title says
which one (`spec-sync V1:` / `spec-sync V2:`).

| Slack message | Fired by | What you do |
| --- | --- | --- |
| `new spec drift → PR opened` (thread root) | `spec-sync.yml`, right after the PR exists | Nothing yet — the AI wiring commit is still being written. |
| `AI wiring …` (result) | `spec-sync.yml`, after the AI step | Review the PR. See §2. |
| `new drift behind the open PR` | `notify-new-drift.sh` | Decide: merge or close. See §3. |
| `gates failed on the sync PR` | `spec-sync-lifecycle.yml` (PR Gates failure) | Fix or close — it cannot merge until green. |
| `sync PR open N days` | daily aging sweep, 16:00 UTC | Clear it. See §4. |
| `PR merged — spec back in sync` | `spec-sync-lifecycle.yml` | Hand off to QA. See §5. |

Silence is also information — but not always: an unavailable staging spec source (unbooked cluster →
404) is a deliberate **no-op with no Slack alert**, so "no messages" can mean either "no drift" or
"staging is unbooked". A staging cluster that stays unbooked for days silently stops all drift
detection. If the channel has been quiet for an unusually long stretch, check that staging is booked
before assuming the spec is stable.

## 2. Reviewing a drift PR

The PR arrives as up to two attributed commits — a mechanical snapshot, then the AI wiring (which
may legitimately be absent for workflow-only drift). `.github/copilot-instructions.md` spells out
what is and is not in review scope; the short version:

- **Do not review** `specs/v2-aide.json`, `specs/v1-ade.json`, or anything under `specs/_generated/`.
  They mirror the upstream spec verbatim and cannot be fixed here.
- **Do review** everything hand-maintained the PR touched: `src/`, `tests/`, `api.md`, `README.md`,
  `docs/`.
- **How much the agent could check its own work differs by loop.** The V1 AI step has a shell
  (`Bash(rye *)`, `Bash(./scripts/*)`) and is told to run `./scripts/format` and `./scripts/lint`
  and fix what they report. The V2 step has **no shell at all** — it works from a precomputed
  `.spec-sync-diff.txt` and can run nothing, so its commit was never lint- or test-verified by the
  agent. Apply more suspicion to a V2 wiring commit. (`ade-typescript` differs again: neither of its
  loops can lint.)
- Check what CI cannot: pyright-strict escapes (`object` + `type: ignore`), optionality that
  disagrees with the spec's `required`, and model-pinned smoke tests. Invented URL paths *are*
  caught for `client.v2` by `check-v2-paths.sh` in the `lint` job — but **not** for V1 resources,
  so check those by hand.
- Cross-SDK parity on hand-written top-level params (`password`, `strict`, `grounding`) has **no
  gate at all** — it has to be diffed against `ade-typescript` by hand. CONTRIBUTING's "V2 request
  fields" section has the rules and the incident that produced them.

## 3. New drift arrives while a sync PR is open

**Why this happens:** `check-drift.sh` compares the live spec against `main`, not against the PR
branch. So while a sync PR is open, every hourly tick still reports drift.
`notify-new-drift.sh` suppresses that noise — it compares the live spec against the *PR branch's*
snapshot, stays silent when the PR already covers it, and otherwise pings once per distinct live
spec (deduped by content hash in a sticky PR comment).

When you get that ping, the open PR is a snapshot of *earlier* drift and is now stale. Two options,
both fine:

**Option A — close the PR, take the next one.** Pick this when the PR has not been reviewed yet, or
when the AI wiring looks wrong and the newer drift probably supersedes it. Nothing is lost: the
next run re-derives everything from the live spec.

**Option B — merge the PR, take the rest next round.** Pick this when the PR is already reviewed
and correct. You ship it as a stepping stone and the next PR covers the remainder.

Either way, the next spec-sync run opens a fresh PR covering what is left. **You do not have to wait
for the top of the hour** — Actions → **Spec Sync** → Run workflow triggers it immediately (the
`concurrency` group serializes it against the cron tick, so a manual dispatch is always safe).

What not to do: leave it open and hope. Drift piles up behind it, the aging nudge starts, and the
eventual PR is a much larger AI diff to review.

## 4. Aging nudge

`scripts/spec-sync/aging-nudge.sh` runs in the daily 16:00 UTC sweep and replies in the PR's thread
once it has been open **3 days** (`MIN_AGE_DAYS`), then at most every **3 days** after that
(`MIN_GAP_DAYS`). It is a reminder, not a gate — nothing auto-closes. Treat a second nudge as a sign
to take Option A in §3.

## 5. After merge: QA, then release

**Merging publishes nothing.** That gap is deliberate — it is the QA window ("staging in, production
out"):

1. **You** merge the sync PR.
2. **You** post in **`#ade-release`** and ping **`@qa_team`** (Slack user group, ~7 QA members) that
   `main` has new SDK surface to validate.
3. **QA** installs merged `main` and tests it against staging, while nothing is published:

   ```sh
   pip install "git+https://github.com/landing-ai/ade-python@main"
   export LANDINGAI_ADE_ENVIRONMENT=staging
   ```

4. The API reaches **production**.
5. A maintainer dispatches Actions → **Release** → Run workflow and picks the bump
   (patch / minor / major). Dispatching **is** the release decision — there is no release PR and no
   second confirmation. The workflow stamps the version, prepends the changelog, pushes
   `release: x.y.z` to `main`, tags it, and creates the GitHub Release, which triggers
   `publish-pypi.yml`.

**A red release is usually not a bug.** The Release workflow's gate 0 is the full V1 + V2 e2e suite
against the **live production API**, and nothing downstream runs if it goes red. Two ordinary causes:
the SDK ships surface that has not reached production yet, or production is having an incident.
Both are the gate doing its job. Shipping during a genuine production incident requires temporarily
removing gate 0 from `release.yml` — a deliberate act, documented at the top of that file.

Both release gates sit before any commit, tag, or GitHub Release exists, so an aborted release
leaves nothing to clean up.

## 6. Secrets and access to take over

All in repo Settings → Secrets and variables → Actions unless noted. The two tokens marked **person**
are currently personal PATs — that is why spec-sync PRs are authored by their owner — and they must
be reissued by whoever takes the repo over.

| Secret | Used by | Notes |
| --- | --- | --- |
| `SPEC_SYNC_TOKEN` | spec-sync: push branch, open PR, PR comments | **person.** Fine-grained PAT scoped to this repo, `Contents: Read and write` + `Pull requests: Read and write`. **Must not be `GITHUB_TOKEN`** — GitHub's anti-recursion rule means PRs it authors do not trigger the gate workflows, so every gate would silently stop running. A GitHub App installation token is the cleaner long-term replacement and needs no other workflow change. |
| `RELEASE_TOKEN` | release.yml: push `release: x.y.z`, tag, GitHub Release | **person.** `Contents: write`, **and its owner must be able to bypass the `main` ruleset** (the ruleset requires PRs; org admins have "always" bypass) or the direct push is rejected. Also must not be `GITHUB_TOKEN`: releases it creates do not trigger `publish-pypi.yml`. |
| `ANTHROPIC_API_KEY` | spec-sync AI step | Billed per run. `--max-turns` caps it. |
| `LANDINGAI_ADE_STAGING_APIKEY` | pr-gates `contract-tests` | Gated behind the **`spec-sync-contract`** Environment. Configure a **required reviewer** there — this job executes AI-authored code with the staging key in env. |
| `LANDINGAI_ADE_PRODUCTION_APIKEY` | `e2e-production.yml` (release gate 0) | **Environment secret on `production-e2e` only — do not also store it as a repo secret**, which every branch can read and which defeats the branch rule. Under that Environment's "Deployment branches and tags", restrict to **`main`**: that rule, not the workflow's `if`, is the real control — a `workflow_dispatch` runs the selected ref's copy of the workflow file, which could drop the `if`. Unlike the staging Environment, do **not** add required reviewers here: `release.yml` blocks on this job, so every release would pause for a manual approval. |
| `SLACK_BOT_TOKEN` | `.github/actions/slack-notify` | `xoxb-…`. Threading exists **only** on this path. Must be in `#ade-sdk-pipeline`. |
| `SLACK_SPEC_SYNC_WEBHOOK` | `spec-sync.yml` only | Incoming webhook, used only when the bot token is empty. Flat, no threading — and **not a general fallback**: `spec-sync-lifecycle.yml` (gate failures, aging nudge, PR merged) passes only the bot token, so a webhook-only setup silently loses those three messages. |
| `LANDINGAI_ADE_PYPI_TOKEN` (or `PYPI_TOKEN`) | `publish-pypi.yml`, `release-doctor.yml` | PyPI publish token. |

Not a secret but part of the handoff: the **required reviewer** on the `spec-sync-contract` Environment is a named person too (`production-e2e` deliberately has none — see its row above).
`production-e2e` Environments are named people too.

## 7. Known gaps — do not mistake these for bugs

- **`release-gate.sh` is not wired.** `scripts/spec-sync/release-gate.sh` implements the
  staging-in/production-out route check (every `(path, method)` in the staging snapshot must exist
  in the production spec) but **nothing calls it** — `release.yml`'s only production gate is the e2e
  suite. Same in `ade-typescript`. Before wiring it, note two real problems: it compares the *whole
  staging snapshot*, not the routes the SDK actually implements, so an unwired compat route that has
  not reached production would block releases; and it is V1-only (both the snapshot path and the
  production URL are hardcoded).
- **V2 workflow parity.** `ade-typescript` ships `client.v2.workflow` / `workflowJobs` as a
  hand-maintained surface (in the tracked spec, excluded from AI wiring). This SDK does **not**
  implement `/v2/workflow*` at all — it is deferred, kept in the baseline, and excluded from the
  reverse check in `check-v2-paths.sh`. The two SDKs are genuinely not at parity here.
- **Cross-SDK alias parity has no gate.** See §2 and CONTRIBUTING's "V2 request fields" section. A
  gate for this was built and then deliberately removed; the prompt and CONTRIBUTING are the only
  enforcement, and comparison is by hand.
