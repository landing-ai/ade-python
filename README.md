<!-- @format -->

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="logo-black.svg">
  <img src="logo-black.svg" alt="LandingAI" width="420" />
</picture>

# Agentic Document Extraction Python Library

[![PyPI version](https://img.shields.io/pypi/v/landingai-ade.svg?label=pypi%20\(stable\))](https://pypi.org/project/landingai-ade/)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
[![License](https://img.shields.io/pypi/l/landingai-ade)](https://pypi.org/project/landingai-ade/)

**[Docs](https://docs.landing.ai) · [Playground](https://ade.landing.ai) · [LandingAI](https://landing.ai)**

</div>

The official Python library for the [LandingAI Agentic Document Extraction (ADE) API](https://ade.landing.ai). Parse PDFs and images into structured, grounded Markdown, then extract typed fields with a JSON Schema or Pydantic model.

- Fully typed requests and Pydantic response models
- Sync and async clients with identical surfaces
- Async jobs with a built-in `wait()` helper for large documents
- Automatic retries with exponential backoff
- Optional `save_to` parameter to write responses to disk

## Installation

```sh
pip install landingai-ade
```

## Set Your API Key

[Generate an API key](https://ade.landing.ai/settings/api-key), then export it as an environment variable. The client reads it automatically.

```sh
export VISION_AGENT_API_KEY=<your-api-key>
```

You can also pass the key directly with `LandingAIADE(apikey=...)`. To keep keys out of source control, use a tool like [python-dotenv](https://pypi.org/project/python-dotenv/).

## Quickstart

Parse a document, then extract structured data from it:

```python
from pathlib import Path
from pydantic import BaseModel, Field
from landingai_ade import LandingAIADE

class Invoice(BaseModel):
    invoice_number: str = Field(description="The invoice number")
    total: str = Field(description="Invoice grand total")

client = LandingAIADE()  # reads VISION_AGENT_API_KEY

# 1. Parse: convert the document to structured Markdown
parsed = client.v2.parse(document=Path("invoice.pdf"))
print(parsed.markdown)

# 2. Extract: pull typed fields out of the Markdown
result = client.v2.extract(schema=Invoice, markdown=parsed.markdown)
print(result.extraction)
```

Use `client.v2` for new projects. It is the current API, powered by the DPT-3 model family. The earlier v1 methods (`client.parse`, `client.extract`, `client.split`, and others) remain fully supported; see [v1 API](#v1-api).

The full method reference for both APIs is in [api.md](api.md); usage guides are at [docs.landing.ai](https://docs.landing.ai).

## Parse

Use `client.v2.parse` to convert a document into Markdown plus a structure tree and grounding (pixel-coordinate bounding boxes for every element). Provide exactly one of `document` (a local file) or `document_url`.

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Parse a local file
parsed = client.v2.parse(
    document=Path("path/to/file.pdf"),
    model="dpt-3-pro-latest",       # optional; defaults to the latest DPT-3 Pro model
    save_to="./output",             # optional; saves as {input_file}_parse_output.json
)

# Or parse a file at a URL
parsed = client.v2.parse(document_url="https://example.com/file.pdf")

print(parsed.markdown)              # full document as Markdown
print(parsed.metadata.page_count)   # pages processed
```

The response is a `V2ParseResponse`:

| Field | Description |
| --- | --- |
| `markdown` | The full document as one Markdown string, in reading order. |
| `structure` | A typed tree (`document` → pages → elements) with element types and character spans into `markdown`. |
| `grounding` | A tree mirroring `structure` that adds pixel-coordinate bounding boxes for each element. |
| `metadata` | Processing details: `page_count`, `failed_pages`, `duration_ms`, and `billing` (credits used). |

If some pages cannot be parsed, the request still succeeds (HTTP 206) and `metadata.failed_pages` lists the pages that failed. If a synchronous parse times out, the client raises `V2SyncTimeoutError`; use [jobs](#process-large-documents-asynchronously-jobs) instead.

## Extract

Use `client.v2.extract` to pull structured fields out of Markdown (typically from a parse response) using a schema. The `schema` parameter accepts a Pydantic `BaseModel` subclass, a `dict`, or a JSON string. Provide exactly one Markdown source: `markdown` or `markdown_url`.

```python
from pydantic import BaseModel, Field
from landingai_ade import LandingAIADE

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")

client = LandingAIADE()

result = client.v2.extract(
    schema=Person,                       # Pydantic model, dict, or JSON string
    markdown=parsed.markdown,            # or markdown_url="https://example.com/doc.md"
    save_to="./output",                  # optional
)

print(result.extraction)                 # {"name": "...", "age": ...}
print(result.extraction_metadata)        # per-field source spans in the Markdown
```

The response is a `V2ExtractResult`:

| Field | Description |
| --- | --- |
| `extraction` | The extracted values, matching your schema. |
| `extraction_metadata` | Mirrors `extraction`; each field carries the character spans in the Markdown that the value came from. |
| `metadata` | Processing details, including credits used. |

By default, unsupported schema fields are pruned and reported. Pass `strict=True` to reject such schemas with an error (HTTP 422) instead.

## Process Large Documents Asynchronously (Jobs)

For documents that take longer than a synchronous request allows, create a job and wait for it. `client.v2.parse_jobs` and `client.v2.extract_jobs` share the same shape: `create`, `get`, `list`, and `wait`.

```python
from pathlib import Path
from landingai_ade import LandingAIADE
from landingai_ade.lib.v2_errors import JobFailedError, JobWaitTimeoutError

client = LandingAIADE()

job = client.v2.parse_jobs.create(
    document=Path("path/to/large_file.pdf"),
    service_tier="standard",   # "standard" (default, lower cost) or "priority" (faster)
)
print(job.job_id, job.status)

# Block until the job finishes (polls with backoff)
try:
    done = client.v2.parse_jobs.wait(job.job_id, timeout=600, raise_on_failure=True)
    print(done.result.markdown[:200])
except JobWaitTimeoutError:
    print("Job did not finish in time; it is still running server-side.")
except JobFailedError as e:
    print(f"Job failed: {e}")
```

Every job method returns a normalized `Job` with `job_id`, `status` (`pending`, `processing`, `completed`, `failed`, or `cancelled`), `progress`, `result`, `error`, and `raw` (the unmodified API envelope, for any field not surfaced on the typed model).

```python
# Poll manually instead of blocking
job = client.v2.parse_jobs.get(job.job_id)

# List jobs, with optional filtering
jobs = client.v2.parse_jobs.list(status="completed", page=0, page_size=10)
for job in jobs:
    print(job.job_id, job.status)
print(jobs.has_more)
```

Extract jobs work the same way: `client.v2.extract_jobs` accepts the same arguments as `client.v2.extract`.

## Async Client

Import `AsyncLandingAIADE` and `await` each call. The async client mirrors the entire sync surface, including `client.v2`.

```python
import asyncio
from pathlib import Path
from landingai_ade import AsyncLandingAIADE

async def main() -> None:
    async with AsyncLandingAIADE() as client:
        parsed = await client.v2.parse(document=Path("path/to/file.pdf"))
        print(parsed.markdown)

asyncio.run(main())
```

For higher concurrency, you can use `aiohttp` as the HTTP backend instead of the default `httpx`:

```sh
pip install landingai-ade[aiohttp]
```

```python
from landingai_ade import AsyncLandingAIADE, DefaultAioHttpClient

async with AsyncLandingAIADE(http_client=DefaultAioHttpClient()) as client:
    ...
```

## Environments

The `environment` argument selects the region. Set it in code or with the `LANDINGAI_ADE_ENVIRONMENT` environment variable.

```python
from landingai_ade import LandingAIADE

client = LandingAIADE(environment="eu")  # "production" (default) or "eu"
```

API keys are per-environment: an EU key works only with `environment="eu"`. To point the client at a mock server or proxy, pass `base_url` (and `v2_base_url` if v2 traffic needs a separate target) or set the `LANDINGAI_ADE_BASE_URL` environment variable.

## v1 API

The v1 methods sit directly on the client.

| Method | What it does |
| --- | --- |
| `client.parse(...)` | Parse a document with the DPT-2 model family. |
| `client.extract(...)` | Extract fields from Markdown. |
| `client.split(...)` | Split a multi-document file into sub-documents by classification. |
| `client.classify(...)` | Classify each page of a document. |
| `client.section(...)` | Generate a hierarchical table of contents. |
| `client.extract_build_schema(...)` | Generate an extraction schema from sample documents. |
| `client.parse_jobs`, `client.extract_jobs` | Async jobs (`create`, `get`, `list`). |

```python
import json
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE()

# Split a combined file into sub-documents
parsed = client.parse(document=Path("statements.pdf"), model="dpt-2-latest")
split = client.split(
    split_class=json.dumps([
        {"name": "Bank Statement", "description": "Summarizes account activity over a period."},
        {"name": "Pay Stub", "description": "Details an employee's earnings for a pay period."},
    ]),
    markdown=parsed.markdown,
    model="split-latest",
)
for s in split.splits:
    print(s.classification, s.pages)
```

## Handling Errors

All errors inherit from `landingai_ade.APIError`.

- Connection problems raise a subclass of `landingai_ade.APIConnectionError`.
- Non-success HTTP status codes (4xx, 5xx) raise a subclass of `landingai_ade.APIStatusError` with `status_code` and `response` properties.

```python
import landingai_ade
from landingai_ade import LandingAIADE

client = LandingAIADE()

try:
    client.v2.parse(document_url="https://example.com/file.pdf")
except landingai_ade.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)
except landingai_ade.RateLimitError:
    print("A 429 status code was received; back off and retry.")
except landingai_ade.APIStatusError as e:
    print(e.status_code)
    print(e.response)
```

| Status Code | Error Type                 |
| ----------- | -------------------------- |
| 400         | `BadRequestError`          |
| 401         | `AuthenticationError`      |
| 403         | `PermissionDeniedError`    |
| 404         | `NotFoundError`            |
| 422         | `UnprocessableEntityError` |
| 429         | `RateLimitError`           |
| >=500       | `InternalServerError`      |
| N/A         | `APIConnectionError`       |

### Retries

Connection errors, 408, 409, 429, and 5xx responses are retried twice by default with exponential backoff. Configure with `max_retries`:

```python
client = LandingAIADE(max_retries=0)                 # default is 2
client.with_options(max_retries=5).v2.parse(...)     # per-request
```

### Timeouts

Requests time out after 8 minutes by default. Configure with `timeout` (a float or an [`httpx.Timeout`](https://www.python-httpx.org/advanced/timeouts/#fine-tuning-the-configuration)):

```python
client = LandingAIADE(timeout=20.0)                  # seconds
client.with_options(timeout=5.0).v2.parse(...)       # per-request
```

On timeout, an `APITimeoutError` is raised. Timed-out requests are retried twice by default.

## Advanced Usage

### Accessing raw response data (e.g. headers)

Prefix any method call with `.with_raw_response.` to get the raw HTTP response:

```python
response = client.with_raw_response.parse(document=Path("file.pdf"), model="dpt-2-latest")
print(response.headers.get("X-My-Header"))
parsed = response.parse()  # the object the method would have returned
```

Use `.with_streaming_response` instead to stream the body rather than reading it eagerly; it requires a context manager and reads the body only when you call `.read()`, `.text()`, `.json()`, `.iter_bytes()`, `.iter_text()`, `.iter_lines()`, or `.parse()`. These return [`APIResponse`](https://github.com/landing-ai/ade-python/tree/main/src/landingai_ade/_response.py) (or `AsyncAPIResponse`) objects.

### Nested params and file uploads

Nested request parameters are [TypedDicts](https://docs.python.org/3/library/typing.html#typing.TypedDict); responses are [Pydantic models](https://docs.pydantic.dev) with helpers such as `model.to_json()` and `model.to_dict()`. File upload parameters accept `bytes`, a [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) instance, or a `(filename, contents, media type)` tuple; the async client reads `PathLike` files asynchronously.

### How to tell whether `None` means `null` or missing

```python
if response.my_field is None:
    if "my_field" not in response.model_fields_set:
        print('The "my_field" key was absent from the response.')
    else:
        print('The response contained "my_field": null.')
```

### Making custom or undocumented requests

Use `client.get` / `client.post` for undocumented endpoints (client options such as retries still apply), and `extra_query`, `extra_body`, or `extra_headers` for undocumented parameters. Undocumented response properties are available via `response.unknown_prop` or [`response.model_extra`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_extra).

### Configuring the HTTP client

Override the [httpx client](https://www.python-httpx.org/api/#client) for proxies, custom transports, or other advanced behavior:

```python
import httpx
from landingai_ade import LandingAIADE, DefaultHttpxClient

client = LandingAIADE(
    http_client=DefaultHttpxClient(
        proxy="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

You can also change it per-request with `client.with_options(http_client=...)`.

### Managing HTTP resources

The client closes HTTP connections when garbage collected. Close it explicitly with `.close()`, or use a context manager:

```python
with LandingAIADE() as client:
    ...  # connections close on exit
```

### Logging

Set the `LANDINGAI_ADE_LOG` environment variable to `info` (or `debug` for more detail):

```sh
export LANDINGAI_ADE_LOG=info
```

## Versioning

This package generally follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions, though certain backwards-incompatible changes may be released as minor versions:

1. Changes that only affect static types, without breaking runtime behavior.
2. Changes to library internals which are technically public but not intended or documented for external use. _(Please open a GitHub issue to let us know if you are relying on such internals.)_
3. Changes that we do not expect to impact the vast majority of users in practice.

We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

To check the version in use at runtime:

```python
import landingai_ade
print(landingai_ade.__version__)
```

## Requirements

Python 3.9 or higher.

## Contributing

See [the contributing documentation](./CONTRIBUTING.md). We welcome [issues](https://www.github.com/landing-ai/ade-python/issues) with questions, bugs, or suggestions.
