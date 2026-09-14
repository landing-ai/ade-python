# V2 (ADE) testing & QA guide

The `client.v2.*` surface targets LandingAI's next-generation ADE gateway
(`api.ade.[env].landing.ai`). This guide covers how the V2 surface is tested and
what to check when the upstream spec (`specs/v2-aide.json`) changes.

## Test layout

| Layer | Location | What it covers |
| --- | --- | --- |
| Response models | `tests/test_v2_types.py` | Deserialization of `V2ParseResponse` / `V2ExtractResult` / `V2BuildSchemaResponse` / `V2GroundResult` and their nested models from plain dicts, including unknown-key tolerance. |
| Job normalization | `tests/test_v2_normalize.py` | `normalize_parse_job` / `normalize_extract_job` / `normalize_build_schema_job`: envelope → unified `Job` (status, timestamps, `result`, `error`). |
| Resource wiring | `tests/api_resources/v2/` | `respx`-mocked HTTP: host routing, multipart/JSON bodies, options serialization, job polling. No network. |
| Live smoke | `tests/contract/test_v2_smoke.py` | End-to-end calls against staging (marked `contract`; skipped unless `LANDINGAI_ADE_STAGING_APIKEY` is set). |

Run the offline suites with `rye run pytest tests/test_v2_types.py
tests/test_v2_normalize.py tests/api_resources/v2` (no credentials needed).

The live smoke suite runs only when `LANDINGAI_ADE_STAGING_APIKEY` is exported:

```bash
LANDINGAI_ADE_STAGING_APIKEY=... rye run pytest tests/contract/test_v2_smoke.py -m contract
```

## Current parse-response shape

`POST /v2/parse` (and the completed `parse_jobs` result) returns a
`V2ParseResponse` with:

- `markdown` -- the full document as one Markdown string.
- `structure` (`V2ParseStructure`) -- the `document → page → element` tree.
  **Every node below the root carries its spatial data inline** in a
  `V2ParseNodeGrounding` object (`grounding`):
  - `page` -- 1-indexed page number.
  - `range` (`V2ParseRange`) -- `{start, end}` code-point offsets into
    `markdown` (`metadata.range_units` names the unit, always
    `"unicode_codepoints"`).
  - `box` (`V2ParseBox`) -- `{xmin, ymin, xmax, ymax}` as `[0, 1]` fractions of
    the page width/height (a page node's box is the full page `{0, 0, 1, 1}`).
  - `confidence` -- an optional `[0, 1]` probability. It is present **only** on
    word-granularity `atomic_grounding` segments (`dpt-3-verity`), where it is the
    lowest per-character OCR confidence in the word -- so a word is only as
    trustworthy as its weakest character. It is `None` on node-level `grounding`
    and on models that ground at line granularity (`dpt-3-pro`). When asserting on
    it, treat missing and `None` alike (`if confidence is not None`).
  - Leaf elements additionally carry `atomic_grounding` -- a list of
    `V2ParseNodeGrounding` segments at whichever granularity the model reads at:
    one entry per visual line for `dpt-3-pro`, one per word (each with its
    `confidence`) for `dpt-3-verity`.
    Each segment reuses the node-grounding shape (`page`, `range`, `box`,
    `confidence`). Omitted when `options.atomic_grounding` is `false`.
- With `options.inline_markdown=true`, the document root, each page, and each
  element also carry their own `markdown` slice.
- `metadata` (`V2ParseMetadata`) -- `job_id`, `model_version`, `page_count`,
  `output_markdown_chars`, `range_units`, `openapi_spec`, `failed_pages`
  (1-indexed), `duration_ms`, and `billing` (`V2ParseBilling`).

The legacy top-level `grounding` tree (`V2ParseGrounding` and friends) is retained
on the model for backward compatibility with older gateway responses; current
responses omit it in favor of the inline `grounding` above.

### Encrypted PDFs (`password`)

`options.password` is a **supported** parse option — earlier snapshots documented
it as unimplemented (any value returned a 422), so a 422 mentioning the password
is no longer the expected outcome of merely supplying one.

- `client.v2.parse(...)` and `client.v2.parse_jobs.create(...)` both take a
  top-level `password=` kwarg. It is folded into `options.password` and sent
  there only (an explicit `options["password"]` wins) — `_build_parse_body` in
  `resources/v2/parse.py` owns that merge, so both routes behave identically.
  Only the 2026-07-13 snapshot also declared a top-level `password` form field;
  sending a second copy would double the secret's exposure and could disagree
  with `options` (`extra_body` overrides raw wire fields one at a time), so the
  duplicate is gone. `ade-typescript` folds it the same way.
- The password must not reach a debug log. `_redact_for_logging` in
  `_base_client.py` redacts it both as a form field and inside the JSON-encoded
  `options` string — parsing JSON-looking strings rather than pattern-matching
  them, since `{"password": ...}` decodes to the same key.
  `tests/test_redact_logging.py` covers it.
- The document is decrypted once at the start of processing and the password is
  not retained with the result, so nothing in `V2ParseResponse` echoes it. There
  is no response-model change to assert.
- The field is PDF-only. Three failures are documented, each a 422 carrying a
  stable `code` in the `ErrorResponse` body (`{code, message}`):
  `password_unsupported_content_type` (password supplied for an image or an
  Office document), `encrypted_pdf_wrong_password`, and
  `encrypted_pdf_password_required` (a locked PDF sent without one).

Testing it: `/v2/parse` declares only `200`, `206` and `422`, so *any* rejection
is a 422 — that much is a spec guarantee and safe to assert live
(`test_parse_sync_password_requires_pdf` in `tests/contract/test_v2_smoke.py`
sends a password with a non-PDF byte payload and asserts the status only).
Asserting a specific `code` needs a controlled response body and belongs in
`tests/api_resources/v2/test_parse.py`
(`test_parse_sync_surfaces_encrypted_pdf_error_code`); do not assert it live, and
do not add an encrypted-PDF fixture to the contract suite — the correct-password
success path cannot be pinned without one, and staging is not guaranteed to have
the field deployed ahead of the snapshot.

## Current extract-response shape

`POST /v2/extract` (and the completed `extract_jobs` result) returns a
`V2ExtractResult` with `extraction`, `extraction_metadata`, `markdown`,
`output_ref` (deprecated; renamed to `schema_violation_error` upstream),
`schema_violation_error` (set when `options.strict` is false and the schema had
fields the model could not extract — the extraction is partial), `warnings`
(non-fatal warnings), and `metadata` (`V2ExtractMetadata`): `job_id`,
`model_version`, `duration_ms`, `doc_id`, `input_markdown_chars`,
`output_extraction_chars`, `credit_usage` (deprecated), `range_units`,
`openapi_spec`, and `billing` (`V2ExtractBilling`). The `input_markdown_chars` /
`output_extraction_chars` char counts moved from `billing` onto `metadata`
upstream; both are retained on `V2ExtractBilling` for backward compatibility.

### Extraction options (`strict`, and the unwired `grounding`)

`/v2/extract` and `/v2/extract/jobs` take a nested `options` object. Unlike parse,
extract does **not** expose `options` as a keyword — the only nested option the SDK
reaches is `strict`, through a hand-written top-level shorthand that folds into
`options.strict`.

The snapshot added a second option, `options.grounding` (default `true`; set it
`false` to skip the grounding stage, which makes every `extraction_metadata` leaf
carry `ranges: null` and returns faster — marked Preview upstream). **It is
deliberately not wired.** It is nested, not a top-level `requestBody` property, so
surfacing it would mean adding a third hand-written alias next to `password` and
`strict`. CONTRIBUTING.md pins those two as a cross-SDK contract shared with
`ade-typescript`, and says in as many words that exposing `options` on extract
"becomes a real tie-break and needs a rule here first" — so adding one is a
maintainer's call, not a spec-sync one. Callers who need it today can send
`extra_body={"options": {"grounding": False}}`; `extra_body` merges at the top level
only, so that replaces the whole `options` object and `strict` has to be repeated
inside it rather than passed as `strict=`.

Nothing about the **response** changed: `grounding=false` only nulls out `ranges`
inside `extraction_metadata`, which `V2ExtractResult` already models as an untyped
`Dict[str, object]`.

The async `extract_jobs.create` also accepts `output_save_url` (async jobs only):
when set, the finished result is delivered to that URL and the completed job
reports `output_url` (on `Job.raw`) instead of an inline `result`. The metadata
receipt (billing included) still rides back on the job and is surfaced as a
`dict` on `Job.metadata` — the delivery moves the content, not the receipt. For
inline jobs `Job.metadata` is `None` and the metadata lives on
`result.metadata` instead. `parse_jobs` behaves the same way.

## Current build-schema-response shape

The V2 build-schema **public surface is currently hidden** — `client.v2.build_schema(...)`
and `client.v2.build_schema_jobs` are not wired. The `V2BuildSchemaResponse` model and
its `normalize_build_schema_job` handling are retained internally, so a build-schema
`Job` result still deserializes to a `V2BuildSchemaResponse` with:

- `extraction_schema` — the generated JSON Schema serialized as a **string** (VTRA
  parity — a string, not an object). Parse it with `json.loads(...)` to get the
  schema dict.
- `metadata` (`V2BuildSchemaMetadata`) — `job_id`, `duration_ms`, `openapi_spec`,
  `filename` / `org_id` / `version` (retained for v1 compatibility, unpopulated in
  this version), a `warnings` list of `V2BuildSchemaWarning` (`{code, msg}`, e.g.
  code `nonconformant_schema`), and `billing` (`V2BuildSchemaBilling`).

## Current ground-response shape

`POST /v2/ground` returns a `V2GroundResult` — a pure, stateless join that maps
each extracted field back to the `structure` blocks it was quoted from:

- `grounding` — a tree mirroring the input `extraction_metadata`: nested objects
  and arrays keep their shape, and each `{value, ranges}` leaf is replaced by the
  list of `structure` blocks its ranges overlap (block ids resolve only against
  the `structure` supplied in the request).
- `metadata` (`V2GroundMetadata`) — `job_id`, `duration_ms`, `openapi_spec`, and
  `billing` (`V2GroundBilling`).

`client.v2.ground(...)` takes `extraction_metadata` and `structure`, each of which
accepts a plain `dict` or a pydantic model (so a parse response's `.structure` can
be passed directly). `/v2/ground` is synchronous-only (no async jobs route).

## Async job envelopes

`normalize_parse_job`, `normalize_extract_job`, and `normalize_build_schema_job`
fold the upstream job envelopes into the unified `Job`. All are tolerant of
field-name drift:

- The parse response lives under `result` (older envelopes used `data`).
- Failures arrive as a structured `error` object (`{code, message}`); older parse
  envelopes used a flat `failure_reason` string. Both map to `Job.error`.
- `created_at` / `completed_at` accept ISO-8601 strings or epoch seconds.
- A top-level `metadata` object (present on `output_save_url` deliveries) is
  passed through to `Job.metadata`; inline jobs leave it `None`.
- Unknown / renamed `status` values fall back to `pending` rather than raising;
  the raw envelope is always preserved on `Job.raw`.

### Listing jobs

`parse_jobs.list` / `extract_jobs.list` take `page`, `page_size` and `status` and
return a `JobList` (`list[Job]` plus `.has_more`, `.org_id`, `.page`,
`.page_size`).

- **`page_size` is sent as `pageSize`.** The snapshot renamed the query parameter
  on `GET /v2/parse/jobs` and `GET /v2/extract/jobs` (the V1 list routes already
  used the camelCase name — see the `PropertyInfo(alias="pageSize")` on
  `types/parse_job_list_params.py`). Only the wire name moved: the `page_size`
  keyword is surface-locked and unchanged, and the *response* envelope still
  returns `page_size`. A wrong name is not an error the gateway reports — it just
  falls back to a default page of 10 — so it is asserted both ways: the mocked
  `test_*_job_list_sends_page_size_as_camel_case` cases check the query string
  directly, and `test_job_lists_honor_page_size` in the contract suite checks live
  staging returns a page of the requested size.
- **`cancelled` is now a documented extract status, on the list route only.**
  `GET /v2/extract/jobs` added it to its status enum; `GET /v2/extract/jobs/{job_id}`
  still declares `pending`/`processing`/`completed`/`failed`, and `/v2/parse/jobs`
  was not changed. No code change was needed — `JobStatus.CANCELLED` already exists
  and `Job.is_terminal` already counts it — but because `wait()` polls the
  single-job GET, an extract wait still only settles on `completed` or `failed`.
  `test_extract_job_list_normalizes_cancelled_status` pins the list-route mapping.

## When the spec changes

1. Read the mechanical diff (`git diff` on `specs/v2-aide.json` and
   `specs/_generated/v2_models.py`), including component-schema-only changes: a
   response field can change only a `$ref`'d component and its generated model.
2. Additive **request** fields become new optional keyword params on the
   corresponding `run` / `create` method — but only **top-level** `requestBody`
   properties. Parse forwards free-form `options` through as a JSON string, so most
   parse-option additions need no code change. Extract does not expose `options`, so
   a new *nested* extract option cannot be wired without adding a hand-written alias,
   which CONTRIBUTING.md reserves for a joint decision with `ade-typescript`; leave
   it unwired and say so in the PR (see the extract-options section above).
3. Additive **response** fields are added to the matching model under
   `src/landingai_ade/types/v2/`. Keep removed/renamed fields in place as optional
   for backward compatibility — the surface is release-locked and response parsing
   is lenient (missing fields default to `None`).
4. Add or extend `respx` tests in `tests/api_resources/v2/` and a live assertion
   in `tests/contract/test_v2_smoke.py`, then update `api.md` and this guide.
