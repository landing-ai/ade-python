# V2 (ADE) testing & QA guide

The `client.v2.*` surface targets LandingAI's next-generation ADE gateway
(`api.ade.[env].landing.ai`). This guide covers how the V2 surface is tested and
what to check when the upstream spec (`specs/v2-aide.json`) changes.

## Test layout

| Layer | Location | What it covers |
| --- | --- | --- |
| Response models | `tests/test_v2_types.py` | Deserialization of `V2ParseResponse` / `V2ExtractResult` / `V2GroundResult` and their nested models from plain dicts, including unknown-key tolerance. |
| Job normalization | `tests/test_v2_normalize.py` | `normalize_parse_job` / `normalize_extract_job` / `normalize_build_schema_job` / `normalize_ground_job`: envelope → unified `Job` (status, timestamps, `result`, `error`). |
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
  - Leaf elements additionally carry `atomic_grounding` -- a list of
    `V2ParseNodeGrounding` segments (visual lines today). Omitted when
    `options.atomic_grounding` is `false`.
- With `options.inline_markdown=true`, the document root, each page, and each
  element also carry their own `markdown` slice.
- `metadata` (`V2ParseMetadata`) -- `job_id`, `model_version`, `page_count`,
  `output_markdown_chars`, `range_units`, `openapi_spec`, `failed_pages`
  (1-indexed), `duration_ms`, and `billing` (`V2ParseBilling`).

The legacy top-level `grounding` tree (`V2ParseGrounding` and friends) is retained
on the model for backward compatibility with older gateway responses; current
responses omit it in favor of the inline `grounding` above.

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

The async `extract_jobs.create` also accepts `output_save_url` (async jobs only):
when set, the finished result is delivered to that URL and the completed job
reports `output_url` (on `Job.raw`) instead of an inline `result`.

## Current build-schema-response shape

`POST /v2/extract/build-schema` (and the completed `build_schema_jobs` result)
returns a `V2BuildSchemaResponse` — it generates or refines an extraction JSON
Schema from source markdown and/or a natural-language prompt:

- `extraction_schema` — the generated JSON Schema serialized as a **string**
  (VTRA parity; the field is a string, not an object). It is suitable for passing
  straight to `client.v2.extract(schema=...)`.
- `metadata` (`V2BuildSchemaMetadata`) — `job_id`, `duration_ms`, `filename`,
  `org_id`, `version`, `openapi_spec`, `billing` (`V2BuildSchemaBilling`), and
  `warnings` (a list of `V2BuildSchemaWarning` `{code, msg}` objects, e.g. code
  `nonconformant_schema`). `filename` and `version` are retained for v1
  compatibility but always `None` in this version.

`client.v2.build_schema(...)` takes `markdowns` (inline content or file text),
`markdown_urls`, `prompt`, and `schema` (an existing schema to iterate on;
accepts a pydantic model, a `dict`, or a JSON string, coerced to a JSON string) —
**at least one** must be provided or the SDK raises `ValueError` client-side.
`build_schema_jobs.create` additionally accepts `service_tier`.

## Current ground-response shape

`POST /v2/ground` (and the completed `ground_jobs` result) returns a
`V2GroundResult` — a pure, stateless join that maps each extracted field back to
the `structure` blocks it was quoted from:

- `grounding` — a tree mirroring the input `extraction_metadata`: nested objects
  and arrays keep their shape, and each `{value, ranges}` leaf is replaced by the
  list of `structure` blocks its ranges overlap (block ids resolve only against
  the `structure` supplied in the request).
- `metadata` (`V2GroundMetadata`) — `job_id`, `duration_ms`, `openapi_spec`, and
  `billing` (`V2GroundBilling`).

`client.v2.ground(...)` takes `extraction_metadata` and `structure`, each of which
accepts a plain `dict` or a pydantic model (so a parse response's `.structure` can
be passed directly). `ground_jobs.create` additionally accepts `output_save_url`.

## Async job envelopes

`normalize_parse_job`, `normalize_extract_job`, `normalize_build_schema_job`, and
`normalize_ground_job` fold the upstream job envelopes into the unified `Job`. All
are tolerant of field-name drift:

- The parse response lives under `result` (older envelopes used `data`).
- Failures arrive as a structured `error` object (`{code, message}`); older parse
  envelopes used a flat `failure_reason` string. Both map to `Job.error`.
- `created_at` / `completed_at` accept ISO-8601 strings or epoch seconds.
- Unknown / renamed `status` values fall back to `pending` rather than raising;
  the raw envelope is always preserved on `Job.raw`.

## When the spec changes

1. Read the mechanical diff (`git diff` on `specs/v2-aide.json` and
   `specs/_generated/v2_models.py`), including component-schema-only changes: a
   response field can change only a `$ref`'d component and its generated model.
2. Additive **request** fields become new optional keyword params on the
   corresponding `run` / `create` method (parse forwards free-form `options`
   through as a JSON string, so most parse-option additions need no code change).
3. Additive **response** fields are added to the matching model under
   `src/landingai_ade/types/v2/`. Keep removed/renamed fields in place as optional
   for backward compatibility — the surface is release-locked and response parsing
   is lenient (missing fields default to `None`).
4. Add or extend `respx` tests in `tests/api_resources/v2/` and a live assertion
   in `tests/contract/test_v2_smoke.py`, then update `api.md` and this guide.
