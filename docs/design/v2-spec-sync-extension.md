<!-- @format -->

# Plan: extend spec-sync to V2 (detection + AI-wiring)

Extends the Problem-3 spec-sync pipeline (today V1-only) to also track the **V2** spec and
AI-wire V2 drift. `client.v2.workflow` was going to be the inaugural proof (the one real V2 gap,
like `extract/jobs` for V1 in #104), but workflow is now **intentionally deferred** — see the
Phase-1/Phase-2 notes below. The loop ships live but quiet; it proves itself on the first future
non-workflow V2 drift.

## Grounded current state (2026-07-13, `main`)

- `spec-sync.yml` is V1-only: `V1_SPEC_URL=api.va.staging.../v1/ade/openapi.json`, branch
  `spec-sync/v1`, AI prompt mirrors `resources/parse_jobs.py`.
- `specs/` holds only `v1-ade.json` + `_generated/v1_models.py`.
- `client.v2` implements **files / parse / extract**; **no workflow**.
- Live V2 spec `aide.staging.landing.ai/openapi.json` exposes `/v2/workflow`,
  `/v2/workflow/jobs`, `/v2/workflow/jobs/{job_id}` = sync run + create-job + get-job (the same
  shape as parse/extract).
- #109 (merged) put V2 contract tests + the surface-lock fix on `main`. `pr-gates.yml`
  `contract-tests` fires on any `spec-sync/*` head branch → **the V2 sync PR gets the live
  staging guardrail for free.**
- Prod V2 spec `aide.landing.ai/openapi.json` returns `200` → a V2 release-gate is feasible.

## Host rules (do not conflate — this bit us once already)

| purpose | host |
|---|---|
| V2 **spec** (drift fetch) | `aide.[env]/openapi.json` (staging: `aide.staging.landing.ai`) |
| V2 **API** (SDK calls + contract tests) | `api.ade.[env]` |
| V2 release-gate (prod spec) | `aide.landing.ai/openapi.json` |

## Design

### Phase 1 — V2 drift detection + mechanical snapshot (low-risk)

- Add a **parallel V2 loop** as a second job in `spec-sync.yml` (keeps the proven V1 loop
  untouched; the AI prompt genuinely differs, so a shared matrix would be awkward):
  - `V2_SPEC_URL=https://aide.staging.landing.ai/openapi.json`, `SYNC_BRANCH=spec-sync/v2`.
  - Snapshot `specs/v2-aide.json`, reference models `specs/_generated/v2_models.py`.
  - **Reuse `check-drift.sh` / `fetch-normalize.sh` / `gen-models.sh` verbatim** — they already
    take (url, path) args; no script changes needed.
  - Same "skip if a `spec-sync/v2` PR is already open" guard.
- **Initial committed baseline = the full live aide spec** (workflow included). `check-drift`
  reports no drift, so the loop stays quiet and fires only on a *future real* spec change.
  > **Note (2026-07-13):** an earlier revision stripped `/v2/workflow*` from the baseline so the
  > loop would immediately draft `client.v2.workflow` as the inaugural proof. That was reverted —
  > workflow is intentionally **deferred** (we don't want to ship it yet). The baseline now
  > includes workflow, and the AI-wiring prompt explicitly excludes the three `/v2/workflow*`
  > routes so a future unrelated V2 drift can't pull them in.
- Snapshot scope: snapshot the **whole** normalized aide spec (simplest; `fetch-normalize.sh`
  unchanged). Phantom-drift risk from the aide host's `/v1` paths is low and every drift PR is
  human-reviewed anyway; revisit with a `/v2`-only `jq` filter if it turns out noisy.

### Phase 2 — V2 AI-wiring (the risky part)

> **Deferred / no inaugural proof.** Workflow was the intended demonstration; it is now
> out of scope. The AI-wiring step below ships **dormant** — it fires only on a future real V2
> spec change (a new field on parse/extract, a new non-workflow route), and proves itself then.
> The design is retained for that eventuality.


- A V2-tailored `claude-code-action` step (edits-only, no git — same anti-injection posture as
  V1), prompted to:
  - Inspect the mechanical diff (mechanical commit adds the `/v2/workflow*` paths + regenerated
    models) and **mirror `src/landingai_ade/resources/v2/extract.py` + `parse.py`**, not the V1
    `parse_jobs.py`.
  - Add `resources/v2/workflow.py`: sync `run()` + `WorkflowJobsResource` (create/get/wait/list),
    dual-host via `_v2_url` (automatic), the **unified `Job`** via a new
    `normalize_workflow_job` in `_normalize.py`, a response type under `types/v2/`, schema
    coercion if the run takes a schema.
  - Register in the `resources/v2/v2.py` container (`workflow`, `workflow_jobs` cached
    properties) + `resources/v2/__init__.py` + the top-level explicit-signature delegators.
  - Add a workflow check to `tests/contract/test_v2_smoke.py` and `tests/api_resources/` tests;
    update `docs/v2-testing.md` + `api.md`.
  - Rules: purely additive; run `./scripts/format` + `./scripts/lint`.
- **Guardrails**: surface-lock (additive-only), **V2 contract tests on the `spec-sync/v2`
  branch** (live), lint/test/typecheck. **Human review required.**
- **Honest caveat**: the V2 ergonomic layer (Job normalization, envelope divergence, `wait`
  semantics, schema coercion) is *not expressed in the spec*, so the AI output is a **reviewed
  draft a human finishes** — heavier finishing than V1. That's expected, not a failure of the loop.

### Phase 3 — V2 release-gate (deferred; NOT in PR 1)

- **Finding while building PR 1:** `scripts/spec-sync/release-gate.sh` (the V1 gate) is **not
  wired into any workflow** — `release.yml` never invokes it, so the "don't ship a staging-only
  route" guard is currently inert for V1 too. Adding a V2 gate "beside the V1 gate" was therefore
  based on a false premise.
- **Decision:** descope release-gate from PR 1. Wiring the release path (for *either* version)
  is a separate change that shouldn't ride along in a detection-loop PR. Tracked as a follow-up:
  parameterize `release-gate.sh` to take `(prod-url, committed-path)`, then invoke it for both
  `specs/v1-ade.json` (vs `api.va.landing.ai/.../openapi.json`) and `specs/v2-aide.json` (vs
  `aide.landing.ai/openapi.json`, confirmed `200`) in `release.yml` before the tag step.

## Risks / open items

- **Phantom drift** from the aide host reordering arrays or churning `/v1` paths → monitor;
  filter to `/v2` if it fires cosmetically.
- **AI-wiring quality** for V2 → mitigated by contract tests + human review; treat as a draft.
- **Workflow contract test** needs a real workflow input; if none is viable, the check is
  documented as skipped rather than faked.

## Build order (subagent-driven, like #99 / #105)

1. **PR 1 — Phase 1** (human-authored, no AI): the V2 loop (second job in `spec-sync.yml`) +
   **full** baseline snapshot `specs/v2-aide.json` + regenerated `specs/_generated/v2_models.py`.
   Small, reviewable, mergeable on its own. On merge the loop is live but quiet (no drift).
   (Phase 3 release-gate deferred — see above.)
2. **No PR 2 for now.** Workflow is deferred, so there is no inaugural AI-wiring PR. The dormant
   AI-wiring step will open its first `spec-sync/v2` PR whenever the V2 spec next changes for a
   non-workflow reason; a human finishes + merges it then.
