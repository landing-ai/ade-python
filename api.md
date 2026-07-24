# Shared Types

```python
from landingai_ade.types import ParseGroundingBox, ParseMetadata
```

# LandingAIADE

Types:

```python
from landingai_ade.types import (
    ClassifyResponse,
    ExtractResponse,
    ExtractBuildSchemaResponse,
    ParseResponse,
    SectionResponse,
    SplitResponse,
)
```

Methods:

- <code title="post /v1/ade/classify">client.<a href="./src/landingai_ade/_client.py">classify</a>(\*\*<a href="src/landingai_ade/types/client_classify_params.py">params</a>) -> <a href="./src/landingai_ade/types/classify_response.py">ClassifyResponse</a></code>
- <code title="post /v1/ade/extract">client.<a href="./src/landingai_ade/_client.py">extract</a>(\*\*<a href="src/landingai_ade/types/client_extract_params.py">params</a>) -> <a href="./src/landingai_ade/types/extract_response.py">ExtractResponse</a></code>
- <code title="post /v1/ade/extract/build-schema">client.<a href="./src/landingai_ade/_client.py">extract_build_schema</a>(\*\*<a href="src/landingai_ade/types/client_extract_build_schema_params.py">params</a>) -> <a href="./src/landingai_ade/types/extract_build_schema_response.py">ExtractBuildSchemaResponse</a></code>
- <code title="post /v1/ade/parse">client.<a href="./src/landingai_ade/_client.py">parse</a>(\*\*<a href="src/landingai_ade/types/client_parse_params.py">params</a>) -> <a href="./src/landingai_ade/types/parse_response.py">ParseResponse</a></code>
- <code title="post /v1/ade/section">client.<a href="./src/landingai_ade/_client.py">section</a>(\*\*<a href="src/landingai_ade/types/client_section_params.py">params</a>) -> <a href="./src/landingai_ade/types/section_response.py">SectionResponse</a></code>
- <code title="post /v1/ade/split">client.<a href="./src/landingai_ade/_client.py">split</a>(\*\*<a href="src/landingai_ade/types/client_split_params.py">params</a>) -> <a href="./src/landingai_ade/types/split_response.py">SplitResponse</a></code>

# ParseJobs

Types:

```python
from landingai_ade.types import ParseJobCreateResponse, ParseJobListResponse, ParseJobGetResponse
```

Methods:

- <code title="post /v1/ade/parse/jobs">client.parse_jobs.<a href="./src/landingai_ade/resources/parse_jobs.py">create</a>(\*\*<a href="src/landingai_ade/types/parse_job_create_params.py">params</a>) -> <a href="./src/landingai_ade/types/parse_job_create_response.py">ParseJobCreateResponse</a></code>
- <code title="get /v1/ade/parse/jobs">client.parse_jobs.<a href="./src/landingai_ade/resources/parse_jobs.py">list</a>(\*\*<a href="src/landingai_ade/types/parse_job_list_params.py">params</a>) -> <a href="./src/landingai_ade/types/parse_job_list_response.py">ParseJobListResponse</a></code>
- <code title="get /v1/ade/parse/jobs/{job_id}">client.parse_jobs.<a href="./src/landingai_ade/resources/parse_jobs.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/parse_job_get_response.py">ParseJobGetResponse</a></code>

# ExtractJobs

Types:

```python
from landingai_ade.types import (
    ExtractJobCreateResponse,
    ExtractJobListResponse,
    ExtractJobGetResponse,
)
```

Methods:

- <code title="post /v1/ade/extract/jobs">client.extract_jobs.<a href="./src/landingai_ade/resources/extract_jobs.py">create</a>(\*\*<a href="src/landingai_ade/types/extract_job_create_params.py">params</a>) -> <a href="./src/landingai_ade/types/extract_job_create_response.py">ExtractJobCreateResponse</a></code>
- <code title="get /v1/ade/extract/jobs">client.extract_jobs.<a href="./src/landingai_ade/resources/extract_jobs.py">list</a>(\*\*<a href="src/landingai_ade/types/extract_job_list_params.py">params</a>) -> <a href="./src/landingai_ade/types/extract_job_list_response.py">ExtractJobListResponse</a></code>
- <code title="get /v1/ade/extract/jobs/{job_id}">client.extract_jobs.<a href="./src/landingai_ade/resources/extract_jobs.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/extract_job_get_response.py">ExtractJobGetResponse</a></code>

# V2

The `client.v2` sub-client targets LandingAI's next-generation ADE gateway, which lives on its own host (`api.ade.[env].landing.ai`) rather than the V1 host (`api.va.[env].landing.ai`). It is **additive**: `client.v2.*` is a separate surface from the top-level `client.*` (V1) methods documented above, and using it does not change any V1 behavior. See the [README](README.md#environments) for environment selection and usage examples.

`client.v2.parse_jobs`, `client.v2.extract_jobs`, `client.v2.build_schema_jobs`, and `client.v2.ground_jobs` all return a single, unified <a href="./src/landingai_ade/types/v2/job.py">`Job`</a> shape, even though the underlying parse/extract/build-schema/ground job envelopes differ upstream -- `Job.raw` retains the full original envelope as an escape hatch for any field not surfaced on the typed model.

Types:

```python
from landingai_ade.types.v2 import (
    Job,
    JobError,
    JobStatus,
    V2BuildSchemaBilling,
    V2BuildSchemaMetadata,
    V2BuildSchemaResponse,
    V2BuildSchemaWarning,
    V2ExtractBilling,
    V2ExtractMetadata,
    V2ExtractResult,
    V2FileUploadResponse,
    V2GroundBilling,
    V2GroundMetadata,
    V2GroundResult,
    V2ParseBilling,
    V2ParseBox,
    V2ParseElement,
    V2ParseGrounding,
    V2ParseGroundingElement,
    V2ParseGroundingEntry,
    V2ParseGroundingPage,
    V2ParseMetadata,
    V2ParseNodeGrounding,
    V2ParsePage,
    V2ParseRange,
    V2ParseResponse,
    V2ParseStructure,
)
```

- <code><a href="./src/landingai_ade/types/v2/job.py">Job</a></code> -- unified job shape: `job_id`, `status` (<code><a href="./src/landingai_ade/types/v2/job.py">JobStatus</a></code>: `pending` / `processing` / `completed` / `failed` / `cancelled`), `created_at`, `completed_at`, `progress`, `result` (a `V2ParseResponse` for parse jobs, a `V2ExtractResult` for extract jobs, a `V2BuildSchemaResponse` for build-schema jobs, or `None` until completion), `error` (<code><a href="./src/landingai_ade/types/v2/job.py">JobError</a></code>), `raw` (the full original envelope as a `dict`), and the `.is_terminal` property.
- <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseResponse</a></code> -- `markdown`, `structure`, `grounding`, `metadata` (<code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseMetadata</a></code>, which nests <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseBilling</a></code> and carries `output_markdown_chars`, `range_units`, and `openapi_spec`). `structure` is a typed <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseStructure</a></code> tree (`document` → <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParsePage</a></code> → <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseElement</a></code>); each node below the root carries its spatial data inline in a <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseNodeGrounding</a></code> (`page`, <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseRange</a></code>, <code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseBox</a></code>, normalized page coordinates), and leaf elements additionally carry an `atomic_grounding` list. With `options.inline_markdown`, each node also carries its `markdown` slice. The legacy top-level `grounding` tree (<code><a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseGrounding</a></code> → `V2ParseGroundingPage` → `V2ParseGroundingElement` → `V2ParseGroundingEntry`) is retained for older gateway responses. Element `type`/page `status` are permissive strings and unknown keys are retained.
- <code><a href="./src/landingai_ade/types/v2/extract_response.py">V2ExtractResult</a></code> -- `extraction`, `extraction_metadata`, `markdown`, `output_ref`, `schema_violation_error` (set when `strict=False` and the schema had unextractable fields), `warnings`, and `metadata` (<code><a href="./src/landingai_ade/types/v2/extract_response.py">V2ExtractMetadata</a></code>, which carries `model_version`, `input_markdown_chars`, `output_extraction_chars`, `range_units`, `openapi_spec`, and nests <code><a href="./src/landingai_ade/types/v2/extract_response.py">V2ExtractBilling</a></code>).
- <code><a href="./src/landingai_ade/types/v2/build_schema_response.py">V2BuildSchemaResponse</a></code> -- `extraction_schema` (the generated JSON Schema serialized as a string) and `metadata` (<code><a href="./src/landingai_ade/types/v2/build_schema_response.py">V2BuildSchemaMetadata</a></code>: `job_id`, `duration_ms`, `openapi_spec`, `filename`/`org_id`/`version` (retained for compatibility), a `warnings` list of <code><a href="./src/landingai_ade/types/v2/build_schema_response.py">V2BuildSchemaWarning</a></code> (`code`, `msg`), and nested <code><a href="./src/landingai_ade/types/v2/build_schema_response.py">V2BuildSchemaBilling</a></code>).
- <code><a href="./src/landingai_ade/types/v2/file_upload_response.py">V2FileUploadResponse</a></code> -- `file_ref`.
- <code><a href="./src/landingai_ade/types/v2/ground_response.py">V2GroundResult</a></code> -- `grounding` (a tree mirroring the input `extraction_metadata`, each `{value, ranges}` leaf replaced by the list of `structure` blocks its ranges overlap) and `metadata` (<code><a href="./src/landingai_ade/types/v2/ground_response.py">V2GroundMetadata</a></code>: `job_id`, `duration_ms`, `openapi_spec`, and nested <code><a href="./src/landingai_ade/types/v2/ground_response.py">V2GroundBilling</a></code>).

Methods:

- <code title="post /v2/parse">client.v2.<a href="./src/landingai_ade/resources/v2/v2.py">parse</a>(\*, document=..., document_url=..., model=..., options=..., password=..., save_to=...) -> <a href="./src/landingai_ade/types/v2/parse_response.py">V2ParseResponse</a></code>

  Synchronous parse. Provide exactly one of `document` (file) or `document_url`. Returns a `V2ParseResponse` on both full success (HTTP 200) and partial success (HTTP 206, where `result.metadata.failed_pages` lists unparsed pages). Raises `V2SyncTimeoutError` (from `landingai_ade.lib.v2_errors`) on a 504; use `parse_jobs` for long-running documents.

- <code title="post /v2/parse/jobs">client.v2.parse_jobs.<a href="./src/landingai_ade/resources/v2/parse.py">create</a>(\*, document=..., document_url=..., model=..., options=..., password=..., output_save_url=..., service_tier=...) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/parse/jobs/{job_id}">client.v2.parse_jobs.<a href="./src/landingai_ade/resources/v2/parse.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/parse/jobs">client.v2.parse_jobs.<a href="./src/landingai_ade/resources/v2/parse.py">list</a>(\*, page=..., page_size=..., status=...) -> JobList[<a href="./src/landingai_ade/types/v2/job.py">Job</a>]</code>
- <code>client.v2.parse_jobs.<a href="./src/landingai_ade/resources/v2/parse.py">wait</a>(job_id, \*, timeout=600, poll_interval=None, raise_on_failure=False) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>

  Blocks, polling `.get(job_id)` with exponential backoff, until the job reaches a terminal status. Raises `JobWaitTimeoutError` if `timeout` seconds elapse first, and `JobFailedError` if `raise_on_failure=True` and the terminal job carries an `error` (not simply every `failed`/`cancelled` status).

- <code title="post /v2/extract">client.v2.<a href="./src/landingai_ade/resources/v2/v2.py">extract</a>(\*, schema, markdown=..., markdown_url=..., model=..., strict=..., save_to=...) -> <a href="./src/landingai_ade/types/v2/extract_response.py">V2ExtractResult</a></code>

  Synchronous extract. `schema` accepts a pydantic `BaseModel` subclass, a `dict`, or a JSON-encoded string -- all are coerced to a JSON Schema object. Provide exactly one of `markdown` or `markdown_url`. `strict=True` rejects schemas with unsupported fields (HTTP 422) instead of silently pruning them. Raises `V2SyncTimeoutError` on a 504; use `extract_jobs` for long-running documents.

- <code title="post /v2/extract/jobs">client.v2.extract_jobs.<a href="./src/landingai_ade/resources/v2/extract.py">create</a>(\*, schema, markdown=..., markdown_url=..., model=..., strict=..., output_save_url=..., service_tier=...) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/extract/jobs/{job_id}">client.v2.extract_jobs.<a href="./src/landingai_ade/resources/v2/extract.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/extract/jobs">client.v2.extract_jobs.<a href="./src/landingai_ade/resources/v2/extract.py">list</a>(\*, page=..., page_size=..., status=...) -> JobList[<a href="./src/landingai_ade/types/v2/job.py">Job</a>]</code>
- <code>client.v2.extract_jobs.<a href="./src/landingai_ade/resources/v2/extract.py">wait</a>(job_id, \*, timeout=600, poll_interval=None, raise_on_failure=False) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>

  Same polling/timeout semantics as `parse_jobs.wait`. Extract jobs have no `cancelled` status, so `raise_on_failure` only ever triggers on `failed`.

- <code title="post /v2/extract/build-schema">client.v2.<a href="./src/landingai_ade/resources/v2/v2.py">build_schema</a>(\*, markdowns=..., markdown_urls=..., prompt=..., schema=...) -> <a href="./src/landingai_ade/types/v2/build_schema_response.py">V2BuildSchemaResponse</a></code>

  Synchronous schema generation. Generate or edit a JSON Schema for extraction from one or more source Markdown documents and/or a natural-language `prompt`, optionally iterating on an existing `schema`. Provide at least one of `markdowns`, `markdown_urls`, `prompt`, or `schema`. `markdowns` entries are each either an inline markdown string or a file upload (`Path`/`bytes`/file object) -- the request is sent as `multipart/form-data` when any entry is a file, and as JSON otherwise. `schema` accepts a pydantic `BaseModel` subclass, a `dict`, or a JSON-encoded string -- all are coerced to a JSON Schema and sent as a JSON-encoded string. Raises `V2SyncTimeoutError` on a 504; use `build_schema_jobs` for long-running inputs.

- <code title="post /v2/extract/build-schema/jobs">client.v2.build_schema_jobs.<a href="./src/landingai_ade/resources/v2/build_schema.py">create</a>(\*, markdowns=..., markdown_urls=..., prompt=..., schema=..., service_tier=...) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/extract/build-schema/jobs/{job_id}">client.v2.build_schema_jobs.<a href="./src/landingai_ade/resources/v2/build_schema.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/extract/build-schema/jobs">client.v2.build_schema_jobs.<a href="./src/landingai_ade/resources/v2/build_schema.py">list</a>(\*, page=..., page_size=..., status=...) -> JobList[<a href="./src/landingai_ade/types/v2/job.py">Job</a>]</code>
- <code>client.v2.build_schema_jobs.<a href="./src/landingai_ade/resources/v2/build_schema.py">wait</a>(job_id, \*, timeout=600, poll_interval=None, raise_on_failure=False) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>

  Same polling/timeout semantics as `parse_jobs.wait`. Build-schema jobs have no `cancelled` status, so `raise_on_failure` only ever triggers on `failed`.

- <code title="post /v2/ground">client.v2.<a href="./src/landingai_ade/resources/v2/v2.py">ground</a>(\*, extraction_metadata, structure) -> <a href="./src/landingai_ade/types/v2/ground_response.py">V2GroundResult</a></code>

  Synchronous ground. Maps each extracted field back to the `structure` blocks it was quoted from by overlapping `extraction_metadata` ranges against every block's inline `grounding.range`. Both `extraction_metadata` (e.g. `client.v2.extract(...).extraction_metadata`) and `structure` (e.g. `client.v2.parse(...).structure`) accept a `dict` or a pydantic model. Block ids resolve only against the `structure` supplied here, so pass the parse result the extraction actually came from. Raises `V2SyncTimeoutError` on a 504; use `ground_jobs` for long-running inputs.

- <code title="post /v2/ground/jobs">client.v2.ground_jobs.<a href="./src/landingai_ade/resources/v2/ground.py">create</a>(\*, extraction_metadata, structure, output_save_url=...) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/ground/jobs/{job_id}">client.v2.ground_jobs.<a href="./src/landingai_ade/resources/v2/ground.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>
- <code title="get /v2/ground/jobs">client.v2.ground_jobs.<a href="./src/landingai_ade/resources/v2/ground.py">list</a>(\*, page=..., page_size=..., status=...) -> JobList[<a href="./src/landingai_ade/types/v2/job.py">Job</a>]</code>
- <code>client.v2.ground_jobs.<a href="./src/landingai_ade/resources/v2/ground.py">wait</a>(job_id, \*, timeout=600, poll_interval=None, raise_on_failure=False) -> <a href="./src/landingai_ade/types/v2/job.py">Job</a></code>

  Same polling/timeout semantics as `parse_jobs.wait`. Ground jobs have no `cancelled` status, so `raise_on_failure` only ever triggers on `failed`. `output_save_url` (create only) delivers the result out-of-band and reports `output_url` on the completed job instead of an inline `result`.

- <code title="post /v1/files">client.v2.files.<a href="./src/landingai_ade/resources/v2/files.py">upload</a>(\*, file) -> str</code>

  Stages a file's bytes on the ADE data plane and returns a `file_ref` string for use in job inputs. Served on the ADE host under `/v1/files` (not `/v2/...`). Raises `LandingAiadeError` if the response has no `file_ref`.

Notes:

- `parse_jobs.list` / `extract_jobs.list` / `build_schema_jobs.list` / `ground_jobs.list` all return a `JobList` (a `list[Job]` subclass) carrying pagination metadata: `.has_more`, `.org_id`, `.page`, `.page_size`.
- All `client.v2.*` methods accept the usual `extra_headers`, `extra_query`, `extra_body`, and `timeout` overrides; sync methods additionally accept `save_to` (parse/extract only, not the job-creation methods) to write the response to disk, mirroring V1's `save_to`.
