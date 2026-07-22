# Copilot instructions — landing-ai/ade-python

## Repository context

This is the LandingAI ADE Python SDK. `client.v2` is a **hand-maintained ergonomic
layer** (unified `Job` polling, dual-host routing, schema coercion) that is intentionally
**not** described by the OpenAPI spec. Generated reference models live under
`specs/_generated/`; the shipped, more-ergonomic surface lives under `src/landingai_ade/`.
When reviewing, do not report the shipped V2 surface as "diverging from the spec" when the
divergence is this deliberate ergonomic layer.

## Reviewing automated spec-sync PRs

PRs authored by `spec-sync[bot]` on branch `spec-sync/v2` arrive as **up to two commits,
in this order**:

1. **Mechanical snapshot** — `chore(spec-sync): update V2 spec snapshot + regenerated
   reference models`. Touches `specs/v2-aide.json` and `specs/_generated/` **only**. By
   design it does **not** wire the client: no new resources, methods, public types,
   exports, tests, or docs. This is expected and correct.
2. **AI wiring** — `feat(spec-sync): wire client.v2 to spec diff (AI)`. Adds the client
   resources / methods / types / exports / tests / docs for the spec diff. This commit
   **may be absent**: workflow-only drift and AI no-ops legitimately produce a
   mechanical-only PR.

When performing a code review on these PRs:

- Review the PR **in its current state, as a whole**. If only the mechanical snapshot
  commit is present, do **not** report the absence of client wiring (missing resources,
  methods, public types, exports, tests, or docs) as an issue — a follow-up wiring commit
  handles it, or the drift is intentionally mechanical-only. Treat the un-wired snapshot as
  complete for now, and do not enumerate the wiring a later commit is expected to add.
- Once the wiring commit is present, focus the review on whether the wired surface matches
  the spec diff: routes, parameter names, request/response fields, required-vs-optional,
  and sync/async (job) surface consistency.
