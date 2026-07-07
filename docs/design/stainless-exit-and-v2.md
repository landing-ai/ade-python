# ADE SDKs: Stainless exit, V2 support, and spec-sync automation

- **Status**: Approved (2026-07-07)
- **Date**: 2026-07-06
- **Scope**: [ade-python](https://github.com/landing-ai/ade-python), [ade-typescript](https://github.com/landing-ai/ade-typescript), one small change in `aide`
- **Context**: Stainless (acquired by Anthropic, 2026-05-18) sunsets its platform for our account on **2026-09-01**.

This design covers three separable problems. They share a deadline and a theme, but each can be discussed, approved, and executed independently:

1. **Getting off Stainless** — mostly cleanup; two concrete breaks to fix.
2. **V2 parse / V2 extract support** — the API contract and the SDK surface that wraps it.
3. **The new pipeline** — spec-sync automation that keeps both SDKs tracking both `openapi.json` specs.

---

## Problem 1 — Getting off Stainless

**The SDKs already work without Stainless, and will keep working.** Both repos are owned by `landing-ai`, the published packages on PyPI/npm have zero runtime dependency on Stainless, and the source builds are self-contained. No user is affected by the sunset. This problem is *not* a migration of code — it is a takeover of the maintenance machinery around the code.

Exactly two things break, both verified:

| Break | Evidence | Fix |
|---|---|---|
| **Releases stop.** Today's release pipeline is split: the Stainless GitHub App runs release-please *externally* — it opens the release PR, and on merge creates the git tag and GitHub Release. The in-repo `publish-pypi.yml` / `publish-npm.yml` workflows then trigger on the release event and run on GitHub runners with our registry tokens — that half is already Stainless-free and survives unchanged. | Every recent tag and GitHub Release in both repos was authored by `stainless-app[bot]`; no release-please workflow exists in-repo. Stale bot release PRs open now: [ade-python#90 "release: 1.13.0"](https://github.com/landing-ai/ade-python/pull/90), [ade-typescript#67 "release: 2.8.0"](https://github.com/landing-ai/ade-typescript/pull/67) (both since April). | Replace only the orchestration half: merge the pending release PRs, then add a self-hosted `release-please-action` workflow (~15 lines; the `release-please-config.json` already in each repo keeps working). Publish workflows and tokens need no change. |
| **The TS build fetches a Stainless-hosted artifact.** | `package.json` pins `"tsc-multi": "https://github.com/stainless-api/tsc-multi/releases/download/v1.1.9/tsc-multi.tgz"`. When the `stainless-api` org is archived/removed, `npm install` breaks. | Vendor the tarball, or repoint to a fork under `landing-ai`. |

One pre-existing failure surfaced while verifying the above, and the takeover inherits it: **every `publish-npm` run since v2.5.0 (April 6) is red.** The root `landingai-ade` package publishes fine (npm is current at 2.7.0; PyPI runs are all green at 1.12.0), but a later step fails — **`landingai-ade-mcp` (the MCP server) has never reached npm**, and the unmerged `fix/npm-oidc-auth` branch suggests a known auth cause. Fix it (or explicitly descope MCP npm publishing) as part of this problem, so that after takeover a green publish run actually means everything shipped.

Everything else is cleanup, no urgency beyond tidiness:

- Remove `stainless-sdks/*` conditionals and the `pkg.stainless.com` upload step from both `ci.yml`s.
- Uninstall the Stainless GitHub App from the org.
- Rewrite `CONTRIBUTING.md` in both repos (currently says code is generated and directs changes to Stainless).
- Retire `.stats.yml` (superseded by Problem 3's committed spec snapshots).
- Bump the deprecated `actions/setup-node@v3` in `publish-npm.yml` (Node 20 action runtimes are removed from GitHub runners on 2026-09-16).
- Optional insurance: download the Stainless workspace bundle before the sunset. Not needed under this design (the V1 spec is public at `https://api.va.landing.ai/v1/ade/openapi.json`; `stainless.yml` only matters for their `stlc` tool), but it is free and the option expires 2026-09-01.

**Exit criteria**: one release published end-to-end from each repo with no Stainless involvement. Estimated effort: 1–2 days total.

### Why not `stlc`

`stlc` is the exit path Stainless offers: a source-available CLI that keeps regenerating SDKs from the OpenAPI spec + `stainless.yml`, re-applying hand-written changes on top via tracking files. Its entire value is continuity — adopting it makes sense **iff we want to keep regenerating from `stainless.yml`**. We don't:

- **The regeneration has nothing left to do.** The surface is small (12 V1 routes, 2 languages) and low-churn, and the backward-compatibility guarantee (Problem 2) means the generated V1 code should barely change again. The sync problem it solves is covered by Problem 3's pipeline, built from maintained, swappable OSS parts.
- **V2 would fight the tool.** Everything we plan to build — `client.v2.*`, job waiters, typed extraction — would live as "custom code" that stlc re-merges onto every regeneration; its own docs warn against restructuring generated output.
- **It is a dead end by design**: feature-frozen, unsupported (no SLA), source-available / internal-use-only, with onboarding help ending 2026-09-01. Adopting it makes our SDK toolchain unmaintained software on day 1 — a new platform dependency, which requirement 3 explicitly avoids.
- **It is heavier than what it replaces**: the recommended setup is a config repo + per-target private staging repos + seal-back PRs + two PATs — more standing infrastructure than the entire Problem 3 pipeline.

It *would* be the right call for a large, fast-moving generated surface or many target languages; neither is planned. The bundle download above keeps this option open until 2026-09-01 — reversible until then, closed after.

---

## Problem 2 — SDK support for V2 parse and V2 extract

### The API contract

Derived from the aide gateway source (`services/gateway/`, `packages/aide_temporal/.../public_workflows/{parse,extract_v2}.py`, `customer_surface.py`). **The exposed `openapi.json` is the source of truth once public — this table must be re-verified against it** (see the aide ask below; the URL currently returns 401).

Auth is unchanged from V1: `Authorization: Bearer <apikey>`. Hosts differ from V1 (see environment matrix below).

**Parse**

| Route | Notes |
|---|---|
| `POST /v2/parse` | Sync. multipart: `document` (file) *or* `document_url`; `model`; `options` (JSON-encoded ParseOptions). Returns `ParseResponse` inline + `X-Version` header. |
| `POST /v2/parse/jobs` | Async, 202. Same fields plus `priority` (`standard`\|`priority`), `output_save_url` (presigned URL for result delivery), `idempotency_key` (same key → same `job_id`). |
| `GET /v2/parse/jobs/{job_id}` | Poll. Envelope: `{job_id, status: pending|processing|completed|failed, started_at, progress, data: ParseResponse, output_url?, metadata, failure_reason?}`. |
| `GET /v2/parse/jobs` | List, paginated (`page`, `page_size`, `status`) → `{jobs, org_id, has_more}`. |

**Extract**

| Route | Notes |
|---|---|
| `POST /v2/extract` | Sync. JSON body: `schema` (JSON Schema, required); markdown from exactly one of `markdown` (inline) \| `markdown_ref` (from `POST /v1/files`) \| `markdown_url`; `model`; `options: {strict}`. Returns `{extraction, extraction_metadata (per-field {value, spans}), markdown, metadata {job_id, version, duration_ms, doc_id?, credit_usage}}`. **206** = partial success under strict mode. |
| `POST /v2/extract/jobs` | Async, 202. Plus `priority`, `idempotency_key`. |
| `GET /v2/extract/jobs/{job_id}` | Poll. Envelope: `{job_id, status, created_at, completed_at?, result?, error? {code, message}, progress?}`. |
| `GET /v2/extract/jobs` | List, paginated. |
| `POST /v1/files` | Staging endpoint for large markdown → `{file_ref}`; referenced via `markdown_ref`. |

Contract items to settle with the aide team (cheap now, expensive after external adoption):

1. **Envelope divergence**: parse jobs use `data`/`output_url`/`started_at`/`failure_reason`; extract jobs use `result`/`error`/`created_at`/`completed_at`. Recommend unifying before GA; otherwise SDKs paper over it forever.
2. `POST /v2/parse` with `password` returns 422 (encrypted PDFs unsupported) — confirm intended for launch.
3. `/v2/workflow` exists in the gateway but is **out of scope** here — confirm.
4. EU availability timing for V2 (`aide.eu-west-1.landing.ai` manifests exist).

### The SDK surface

Additive only — Python stays 1.x, TS stays 2.x, existing methods and types are untouched (V2 types live in their own `types/v2/` module and V1 types are never regenerated). V2 lands under a `v2` sub-client that shares the transport (auth, retries, http client) and carries its own base URL:

```python
client = LandingAIADE(apikey=...)                         # prod, exactly as today
client = LandingAIADE(apikey=..., environment="staging")  # QA

client.parse(document=...)                    # unchanged → {v1}/v1/ade/parse
client.v2.parse(document=..., model=...)      # new → {v2}/v2/parse
job = client.v2.parse_jobs.create(document=..., priority="priority")
res = client.v2.parse_jobs.wait(job.job_id, timeout=600)   # polling helper
client.v2.extract(schema=InvoiceModel, markdown_url=...)   # pydantic/zod schema accepted
client.v2.files.upload(...)                   # → file_ref for markdown_ref
```

Hand-written ergonomics on top of the generated types: `wait()` with backoff, pydantic/zod models accepted as `schema` (extending the existing `lib/schema_utils.py`), one `Job` shape presented to users regardless of the envelope question above, `idempotency_key` pass-through, `save_to` support on V2 methods for parity.

### Multi-environment

The V2 API lives on different hosts, so the existing 2-entry `environment` map becomes a 4-entry map of host *pairs*, and `environment` is the entire user-facing story — the client resolves both hosts from it and the `v2` sub-client routes to the right one internally:

| `environment` | V1 base URL | V2 base URL |
|---|---|---|
| `production` (default) | `https://api.va.landing.ai` | `https://aide.landing.ai` |
| `eu` | `https://api.va.eu-west-1.landing.ai` | `https://aide.eu-west-1.landing.ai` |
| `staging` | `https://api.va.staging.landing.ai` | `https://aide.staging.landing.ai` |
| `dev` | `https://api.va.dev.landing.ai` | `https://aide.dev.landing.ai` |

A new `LANDINGAI_ADE_ENVIRONMENT` env var selects the environment without code changes — that's the QA workflow. Explicit `base_url` / `v2_base_url` overrides (params and env vars) remain for mock servers and proxies; if only `base_url` is set, V2 traffic follows it too, so a mock captures everything. Two implementation subtleties are deferred to the PR, noted here only so they aren't rediscovered: routing is per-resource rather than by path prefix (`POST /v1/files` lives on the aide host), and V1 request paths are untouched, so existing behavior can't change. API keys stay per-environment, passed through as today. Open item: security sign-off on shipping `dev`/`staging` hostnames in public source (not secrets; fallback is env-var-only recognition).

### Required from aide (one change)

Serve the curated, customer-surface-only OpenAPI spec unauthenticated at each environment's gateway host — `https://aide[.env].landing.ai/openapi.json` currently returns 401 (staff SSO). The curation hook (`install_job_openapi()`) already exists; verify staff routes are excluded and multipart bodies render correctly for codegen. This unblocks both the contract verification above and Problem 3.

---

## Problem 3 — The pipeline: spec-sync automation

Goal: no codegen/release platform, but the SDKs mechanically track **both** live specs — `https://api.va.landing.ai/v1/ade/openapi.json` (V1) and `https://aide.landing.ai/openapi.json` (V2). Drift is real today: the live V1 spec has 3 `/v1/ade/extract/jobs` routes that neither SDK implements.

One workflow per SDK repo (`.github/workflows/spec-sync.yml`), triggered by cron (~6h) + manual dispatch (+ optional `repository_dispatch` from aide deploys later — cron covers correctness, dispatch only cuts latency):

```
fetch v1 + v2 openapi.json ─ normalize (jq -S) ─ diff vs committed specs/*.json
  └─ on drift, ONE job, two phases, one PR branch:
       commit 1 (mechanical): spec snapshot + regenerated types
                              (datamodel-code-generator / openapi-typescript)
       commit 2 (AI):         Claude Code GitHub Action wires resources,
                              methods, tests, docs from the spec diff
  └─ PR gates (ordinary CI jobs):
       surface-lock: api-extractor report (TS) / griffe check (Python)
         → any change to RELEASED public surface fails mechanically
           (baseline = last release tag; see spec-targeting below)
       contract tests vs staging (uses Problem 2's environment support)
       existing mock tests · lint · typecheck
  └─ human review → merge → self-hosted release-please → publish on release
```

Design notes:

- **One job, two phases**: the AI step needs the freshly generated types in its working tree, and a codegen failure must abort before the AI runs. Two separately attributed commits give reviewers an audit boundary — skim the mechanical commit, scrutinize the AI commit.
- **Committed snapshots** (`specs/v1-ade.json`, `specs/v2-aide.json`) make every API change visible in the PR diff. This replaces `.stats.yml`.
- **Per-repo and independent**: each SDK repo carries its own copy (same shape, different codegen tool). aide only serves the spec; no tokens flow upstream.
- The surface-lock gate is what makes the AI step safe to run unattended: the backward-compatibility guarantee from Problem 2 is enforced by CI, not by convention or review vigilance.

### Which environment's spec drives the loop: staging in, production out

The automation targets the **staging** spec; releases gate on the **production** spec. This decouples "SDK code is merged" from "SDK artifact is released", which is exactly the QA window we want:

1. API deploys to staging → staging spec changes → sync PR opens (or refreshes its branch).
2. Automated gates run against staging (contract tests, surface-lock) → review → merge to `main`.
3. QA tests merged `main` against staging (`pip install git+…@main` + `LANDINGAI_ADE_ENVIRONMENT=staging`) while the artifact is *not yet published*.
4. API deploys to production → the release gate (every SDK-implemented path exists in the production spec) passes → release-please PR merges → PyPI/npm artifact ships.

Two consequences to be explicit about:

- **The surface-lock baseline must be the last *release tag*, not the previous commit on `main`.** Merged-but-unreleased surface stays mutable: if a staging feature is reshaped or dropped before it ever reaches production, the next sync PR can amend or remove it without tripping the compat gate. Only *released* surface is locked — which is the actual promise made to users.
- **A staging-only feature holds the release train**: nothing releases until it either reaches production or is reverted in staging (at which point the next sync PR removes it and unblocks). Self-correcting in both directions, and acceptable at our deploy cadence.

Verified: the V1 staging spec is already public (`https://api.va.staging.landing.ai/v1/ade/openapi.json` → 200); aide's staging spec is behind the same auth as production, so the single aide ask in Problem 2 (expose the curated spec per environment) covers this.

### The AI step, concretely: `anthropics/claude-code-action`

This is the tool the Stainless transition guide itself names for post-Stainless SDK maintenance. What it is: an MIT-licensed, Anthropic-maintained GitHub Action ([repo](https://github.com/anthropics/claude-code-action), [docs](https://code.claude.com/docs/en/github-actions)) that runs the Claude Code agent headlessly on our own runner. v1 has been GA since Aug 2025. It has two auto-detected modes — interactive (responds to `@claude` mentions on PRs/issues) and automation (executes a `prompt` input non-interactively). The pipeline uses **automation mode only**: the step runs inside the spec-sync job after the mechanical phase, on the same checkout/branch, edits files per the prompt, and commits — producing phase-2's separately attributed commit.

Operational facts that shape the workflow:

- **Auth/cost**: our Anthropic API key as a repo secret (`anthropic_api_key`), or OIDC to Bedrock/Vertex if preferred. Code never leaves the runner except calls to the model API. Cost is per-run model usage; `--max-turns` caps it.
- **Guardrails are inputs, not hopes**: `claude_args: --allowedTools ...` restricts what it can execute; the prompt states the contract ("never modify existing public surface — CI will fail you"); it cannot merge; branch protection and human review apply as with any PR.
- **Token gotcha**: commits pushed with the default `GITHUB_TOKEN` do **not** trigger other workflows (GitHub's anti-recursion rule) — so the gate jobs would never run on the sync PR. The job must push with a GitHub App installation token or fine-grained PAT (`contents: write`, `pull-requests: write` on the SDK repo). Same pattern for the mechanical phase's push.
- **No lock-in**: the step is a thin wrapper over a headless agent CLI. If we ever prefer a different agent, the pipeline shape (mechanical commit → agent commit → gates) is agent-agnostic.

Illustrative step (pin `@v1`; final flags at implementation time):

```yaml
- uses: anthropics/claude-code-action@v1
  with:
    anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    github_token: ${{ steps.app-token.outputs.token }}
    prompt: |
      The previous commit updated specs/*.json and the generated types to match.
      Update resource classes, method signatures, tests, api.md and README
      examples accordingly. Rules: purely additive — never modify existing
      public signatures (the surface-lock CI job will fail the PR); follow
      existing code conventions; commit to the current branch.
    claude_args: |
      --max-turns 30
      --allowedTools "Edit,Write,Read,Bash(git *),Bash(./scripts/*)"
```

**Proving the loop**: stand it up in ade-python first and let its inaugural PR be the known V1 gap (`extract/jobs` ×3). [PR #93](https://github.com/landing-ai/ade-python/pull/93) already hand-starts this — reconcile rather than duplicate. This validates the pipeline end-to-end on a real change before any V2 code exists, then port the workflow to ade-typescript.

---

## Suggested order & ownership

Problem 1 (July, ~2 days) → Problem 3 proven on V1 (July) → Problem 2 once aide exposes the spec (Aug), with the pipeline then extending to V2 automatically. Open: who owns sync-PR review and release cadence; staging API keys for CI (which vault); `/v2/workflow` scope confirmation.
