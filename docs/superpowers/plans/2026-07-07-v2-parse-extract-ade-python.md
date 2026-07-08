# V2 Parse & Extract SDK Support (ade-python) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive `client.v2` sub-client to ade-python exposing V2 parse, V2 extract, their async job routes, and file staging — routed to the ADE (aide) host — with hand-written ergonomics (unified `Job` shape, `wait()` polling, pydantic-schema acceptance, `save_to` parity).

**Architecture:** The existing `LandingAIADE` transport (auth, retries, http client) is reused unchanged. `client.v2` is a container resource holding sub-resources that build **absolute** URLs against a second base URL (the V2/ADE host). The transport's `_prepare_url` passes absolute URLs through untouched, so no new HTTP client is needed. V2 request/response types are hand-authored pydantic models under `types/v2/`; the two divergent job envelopes (parse vs extract) are normalized into one `Job` model. Ergonomics live in new focused modules; edits to Stainless-generated files are minimal and additive.

**Tech Stack:** Python 3.9+, httpx, pydantic (v1 **and** v2 must both work), pytest + respx for HTTP mocking, anyio for async.

## Global Constraints

- **Additive only.** No existing V1 method, type, path, or signature changes. Package stays `1.x`. Every V1 request path (`/v1/ade/*`) is byte-for-byte unchanged.
- **Both pydantic majors.** `pydantic>=1.9,<3`. Subclass the SDK's own `landingai_ade._models.BaseModel` for new models (it handles v1/v2 compat, `.to_json()`, extra-field capture). Do not call pydantic-v2-only APIs directly.
- **Python floor 3.9.** No `match`, no `X | Y` runtime unions in annotations that execute (use `from __future__ import annotations` in every new module, as the repo does), no `list[str]` at runtime in 3.9-incompatible positions.
- **Generated-file edits are limited and additive**, confined to: `_client.py` (environment map, `v2_base_url` resolution, one `v2` cached property on each client, `environment` Literal widened), `types/__init__.py` / top-level `__init__.py` (re-exports). All other V2 code lives in **new** files under `types/v2/`, `resources/v2/`, and `lib/`, which no generator manages. This holds whether or not the Stainless exit (design doc "Problem 1") has landed.
- **Contract source of truth:** the verified OpenAPI at `/private/tmp/claude-501/-Users-zhichao-Projects-aide/6f31597c-25a3-4357-a696-ad95d591b156/scratchpad/openapi.json`. Verified deltas from the design doc that this plan encodes: `idempotency_key` is **extract-only**; **206** partial-success is on **parse** sync (not extract); encrypted-PDF `password` is **supported** on parse; parse job timestamps are **int epoch seconds** while extract's are **ISO strings**; parse failure is `failure_reason`, extract failure is `error{code,message}`; extract has no `cancelled` status; `/v1/files` lives on the **V2/ADE host**; `/v2/workflow` is **out of scope** but the resource/waiter machinery must not preclude adding it later.
- **Environment matrix** (verbatim):

  | environment | V1 base URL | V2 base URL |
  |---|---|---|
  | production (default) | `https://api.va.landing.ai` | `https://aide.landing.ai` |
  | eu | `https://api.va.eu-west-1.landing.ai` | `https://aide.eu-west-1.landing.ai` |
  | staging | `https://api.va.staging.landing.ai` | `https://aide.staging.landing.ai` |
  | dev | `https://api.va.dev.landing.ai` | `https://aide.dev.landing.ai` |

  > **V2 host (corrected 2026-07-08):** the V2 customer surface — API **and** spec — is served by the AIDE gateway at `aide.[env].landing.ai` (`aide.[env]/openapi.json` = title "AIDE Gateway", the 10-path V2 surface). An earlier draft used `api.ade.[env].landing.ai`; that host is the VTRA "Vision Tools API" (V1 `/v1/ade/*` + `/v1/tools/*`) and does **not** serve `/v2/parse|extract|workflow` or `/v1/files`. Confirmed against the two hosts' `openapi.json` content (probe status codes on these POST routes are unreliable — trust the spec).

- **Env vars:** `LANDINGAI_ADE_ENVIRONMENT` selects the environment; `LANDINGAI_ADE_BASE_URL` overrides V1; `LANDINGAI_ADE_V2_BASE_URL` overrides V2. If only `base_url` (V1) is set with no V2 override, **V2 traffic follows `base_url`** (so a single mock captures everything).
- **Auth unchanged:** `Authorization: Bearer <apikey>`, applied globally by the transport to both hosts. Same key per environment.

---

## File Structure

**New files**
- `src/landingai_ade/types/v2/__init__.py` — re-exports all V2 types.
- `src/landingai_ade/types/v2/job.py` — `JobStatus`, `JobError`, `Job`.
- `src/landingai_ade/types/v2/parse_response.py` — `V2ParseResponse`, `V2ParseMetadata`, `V2ParseBilling`.
- `src/landingai_ade/types/v2/extract_response.py` — `V2ExtractResult`, `V2ExtractMetadata`.
- `src/landingai_ade/types/v2/file_upload_response.py` — `V2FileUploadResponse`.
- `src/landingai_ade/lib/schema_utils.py` — **extend** with `coerce_schema_to_dict`.
- `src/landingai_ade/lib/v2_errors.py` — `V2SyncTimeoutError`, `JobWaitTimeoutError`, `JobFailedError`.
- `src/landingai_ade/resources/v2/__init__.py` — re-exports `V2Resource`, `AsyncV2Resource`.
- `src/landingai_ade/resources/v2/_normalize.py` — `normalize_parse_job`, `normalize_extract_job`.
- `src/landingai_ade/resources/v2/_base.py` — `V2ResourceMixin` (absolute-URL builder), sync/async `_wait` helpers.
- `src/landingai_ade/resources/v2/files.py` — `FilesResource`, `AsyncFilesResource`.
- `src/landingai_ade/resources/v2/parse.py` — `ParseResource`, `AsyncParseResource`, `ParseJobsResource`, `AsyncParseJobsResource`.
- `src/landingai_ade/resources/v2/extract.py` — `ExtractResource`, `AsyncExtractResource`, `ExtractJobsResource`, `AsyncExtractJobsResource`.
- `src/landingai_ade/resources/v2/v2.py` — `V2Resource`, `AsyncV2Resource` containers.
- `tests/test_v2_environment.py`, `tests/test_v2_types.py`, `tests/test_v2_normalize.py`, `tests/test_v2_schema.py`, `tests/api_resources/v2/test_parse.py`, `tests/api_resources/v2/test_extract.py`, `tests/api_resources/v2/test_files.py`, `tests/api_resources/v2/__init__.py`.

**Modified files**
- `src/landingai_ade/_client.py` — env map → 4 pairs; `v2_base_url` param + resolution; `_v2_base_url` attr; `v2` cached property (sync + async); widen `environment` Literal in `__init__` and `copy`.
- `src/landingai_ade/types/__init__.py` — no change needed (V2 types are imported from `landingai_ade.types.v2`).
- `src/landingai_ade/lib/__init__.py` — export `coerce_schema_to_dict`.
- `api.md`, `README.md`, `examples/` — V2 docs.

---

## Task 1: Environment map + dual base-URL resolution

**Files:**
- Modify: `src/landingai_ade/_client.py` (ENVIRONMENTS, `__init__`, `copy`, both classes)
- Test: `tests/test_v2_environment.py`

**Interfaces:**
- Produces: `LandingAIADE(...)._v2_base_url: str` and `AsyncLandingAIADE(...)._v2_base_url: str` — the resolved V2 host, no trailing slash. New kwarg `v2_base_url: str | None`. `environment` accepts `"production"|"eu"|"staging"|"dev"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v2_environment.py
from __future__ import annotations

import pytest

from landingai_ade import LandingAIADE

APIKEY = "My Apikey"


def test_default_production_pair() -> None:
    c = LandingAIADE(apikey=APIKEY)
    assert str(c.base_url).rstrip("/") == "https://api.va.landing.ai"
    assert c._v2_base_url == "https://aide.landing.ai"


@pytest.mark.parametrize(
    "env,v1,v2",
    [
        ("production", "https://api.va.landing.ai", "https://aide.landing.ai"),
        ("eu", "https://api.va.eu-west-1.landing.ai", "https://aide.eu-west-1.landing.ai"),
        ("staging", "https://api.va.staging.landing.ai", "https://aide.staging.landing.ai"),
        ("dev", "https://api.va.dev.landing.ai", "https://aide.dev.landing.ai"),
    ],
)
def test_environment_pairs(env: str, v1: str, v2: str) -> None:
    c = LandingAIADE(apikey=APIKEY, environment=env)
    assert str(c.base_url).rstrip("/") == v1
    assert c._v2_base_url == v2


def test_environment_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANDINGAI_ADE_ENVIRONMENT", "staging")
    c = LandingAIADE(apikey=APIKEY)
    assert c._v2_base_url == "https://aide.staging.landing.ai"


def test_explicit_v2_base_url_wins() -> None:
    c = LandingAIADE(apikey=APIKEY, v2_base_url="https://mock.local")
    assert c._v2_base_url == "https://mock.local"


def test_v2_follows_base_url_when_only_base_url_set() -> None:
    c = LandingAIADE(apikey=APIKEY, base_url="http://127.0.0.1:4010")
    assert str(c.base_url).rstrip("/") == "http://127.0.0.1:4010"
    assert c._v2_base_url == "http://127.0.0.1:4010"


def test_v2_base_url_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANDINGAI_ADE_V2_BASE_URL", "https://v2.mock.local")
    c = LandingAIADE(apikey=APIKEY)
    assert c._v2_base_url == "https://v2.mock.local"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `rye run pytest tests/test_v2_environment.py -v`
Expected: FAIL (`AttributeError: '_v2_base_url'`, or `TypeError: unexpected keyword 'v2_base_url'`).

- [ ] **Step 3: Implement the environment map + resolution**

In `src/landingai_ade/_client.py`, replace the `ENVIRONMENTS` map:

```python
ENVIRONMENTS: Dict[str, str] = {
    "production": "https://api.va.landing.ai",
    "eu": "https://api.va.eu-west-1.landing.ai",
    "staging": "https://api.va.staging.landing.ai",
    "dev": "https://api.va.dev.landing.ai",
}

V2_ENVIRONMENTS: Dict[str, str] = {
    "production": "https://aide.landing.ai",
    "eu": "https://aide.eu-west-1.landing.ai",
    "staging": "https://aide.staging.landing.ai",
    "dev": "https://aide.dev.landing.ai",
}
```

Add a module-level resolver (place it above the `LandingAIADE` class, after `_save_response`):

```python
def _resolve_v2_base_url(
    environment: str | NotGiven,
    v2_base_url: str | None | NotGiven,
    resolved_v1_base_url: str | httpx.URL,
    v1_base_url_was_explicit: bool,
) -> str:
    """Resolve the V2/ADE host, no trailing slash.

    Precedence: explicit v2_base_url param > LANDINGAI_ADE_V2_BASE_URL env >
    environment map > (if only V1 base_url was set explicitly) follow it >
    production default.
    """
    if is_given(v2_base_url) and v2_base_url is not None:
        return str(v2_base_url).rstrip("/")

    env_override = os.environ.get("LANDINGAI_ADE_V2_BASE_URL")
    if env_override:
        return env_override.rstrip("/")

    if is_given(environment):
        try:
            return V2_ENVIRONMENTS[environment]
        except KeyError as exc:
            raise ValueError(f"Unknown environment: {environment}") from exc

    if v1_base_url_was_explicit:
        # Only a V1 base_url was provided (mock/proxy) — route V2 to it too.
        return str(resolved_v1_base_url).rstrip("/")

    return V2_ENVIRONMENTS["production"]
```

In `LandingAIADE.__init__`, widen the `environment` annotation and add `v2_base_url`:

```python
    _environment: Literal["production", "eu", "staging", "dev"] | NotGiven

    def __init__(
        self,
        *,
        apikey: str | None = None,
        environment: Literal["production", "eu", "staging", "dev"] | NotGiven = not_given,
        base_url: str | httpx.URL | None | NotGiven = not_given,
        v2_base_url: str | None | NotGiven = not_given,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
    ) -> None:
```

Before the `environment` resolution block, allow `LANDINGAI_ADE_ENVIRONMENT` to seed `environment`:

```python
        if not is_given(environment):
            env_name = os.environ.get("LANDINGAI_ADE_ENVIRONMENT")
            if env_name:
                environment = env_name  # type: ignore[assignment]
```

Track whether V1 base_url was explicit (for the "V2 follows base_url" rule). After the existing V1 resolution block computes `base_url`, add:

```python
        v1_base_url_was_explicit = (is_given(base_url) and base_url is not None) or (
            os.environ.get("LANDINGAI_ADE_BASE_URL") is not None and not is_given(environment)
        )
        self._v2_base_url = _resolve_v2_base_url(
            environment, v2_base_url, base_url, v1_base_url_was_explicit
        )
```

> **Note:** place the `self._v2_base_url = ...` assignment *before* `super().__init__(...)` if it references only locals, or after — either works since it only reads `base_url`. Keep it right before `super().__init__`.

Widen the `environment` annotation in `copy` the same way (`Literal["production", "eu", "staging", "dev"] | None`) and thread `v2_base_url` through it:

```python
        v2_base_url: str | None = None,
        ...
        return self.__class__(
            ...,
            v2_base_url=v2_base_url or self._v2_base_url,
            ...,
        )
```

Repeat all of the above verbatim in `AsyncLandingAIADE` (same file, the async class).

- [ ] **Step 4: Run tests to verify they pass**

Run: `rye run pytest tests/test_v2_environment.py -v`
Expected: PASS (all 6+ params).

- [ ] **Step 5: Verify V1 is untouched**

Run: `rye run pytest tests/test_client.py -q`
Expected: PASS (no regressions in existing client tests).

- [ ] **Step 6: Commit**

```bash
git add src/landingai_ade/_client.py tests/test_v2_environment.py
git commit -m "feat(v2): resolve dual V1/V2 base URLs from environment"
```

---

## Task 2: V2 response & job types

**Files:**
- Create: `src/landingai_ade/types/v2/__init__.py`, `job.py`, `parse_response.py`, `extract_response.py`, `file_upload_response.py`
- Test: `tests/test_v2_types.py`

**Interfaces:**
- Produces:
  - `JobStatus(str, Enum)`: `PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED`.
  - `JobError(BaseModel)`: `code: Optional[str]`, `message: Optional[str]`.
  - `Job(BaseModel)`: `job_id: str`, `status: JobStatus`, `created_at: Optional[datetime]`, `completed_at: Optional[datetime]`, `progress: Optional[float]`, `result: Optional[object]`, `error: Optional[JobError]`, `raw: Dict[str, object]`; property `is_terminal: bool`.
  - `V2ParseResponse(BaseModel)`: `markdown: Optional[str]`, `structure: Optional[object]`, `grounding: Optional[object]`, `metadata: Optional[V2ParseMetadata]`.
  - `V2ExtractResult(BaseModel)`: `extraction: Dict[str, object]`, `extraction_metadata: Dict[str, object]`, `markdown: str`, `metadata: V2ExtractMetadata`.
  - `V2FileUploadResponse(BaseModel)`: `file_ref: Optional[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v2_types.py
from __future__ import annotations

from datetime import datetime

from landingai_ade.types.v2 import (
    Job,
    JobError,
    JobStatus,
    V2ExtractResult,
    V2ParseResponse,
    V2FileUploadResponse,
)


def test_job_status_enum_values() -> None:
    assert JobStatus.COMPLETED.value == "completed"
    assert set(JobStatus) >= {
        JobStatus.PENDING,
        JobStatus.PROCESSING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }


def test_job_is_terminal() -> None:
    running = Job(job_id="j1", status=JobStatus.PROCESSING)
    done = Job(job_id="j1", status=JobStatus.COMPLETED)
    failed = Job(job_id="j1", status=JobStatus.FAILED)
    assert running.is_terminal is False
    assert done.is_terminal is True
    assert failed.is_terminal is True


def test_job_holds_typed_result_and_error() -> None:
    job = Job(
        job_id="j1",
        status=JobStatus.FAILED,
        created_at=datetime(2026, 1, 1),
        error=JobError(code="internal_error", message="boom"),
        raw={"org_id": "o1"},
    )
    assert job.error is not None and job.error.code == "internal_error"
    assert job.raw["org_id"] == "o1"


def test_extract_result_parses_nested_metadata() -> None:
    r = V2ExtractResult(
        extraction={"revenue": "1M"},
        extraction_metadata={"revenue": {"value": "1M", "spans": []}},
        markdown="# doc",
        metadata={"job_id": "j1", "version": "extract-1", "duration_ms": 12},
    )
    assert r.metadata.job_id == "j1"
    assert r.metadata.credit_usage == 0.0  # default


def test_parse_response_tolerates_loose_shape() -> None:
    r = V2ParseResponse(
        markdown="# hi",
        structure=[{"type": "text"}],
        metadata={
            "req_id": "r1",
            "job_id": "j1",
            "model_version": "dpt-3",
            "page_count": 2,
            "failed_pages": [],
            "billing": {"service_tier": "standard", "total_credits": 3.0},
        },
    )
    assert r.markdown == "# hi"
    assert r.metadata is not None and r.metadata.billing is not None
    assert r.metadata.billing.total_credits == 3.0


def test_file_upload_response() -> None:
    assert V2FileUploadResponse(file_ref="abc").file_ref == "abc"
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/test_v2_types.py -v`
Expected: FAIL (`ModuleNotFoundError: landingai_ade.types.v2`).

- [ ] **Step 3: Create the type modules**

```python
# src/landingai_ade/types/v2/job.py
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional
from datetime import datetime

from ..._models import BaseModel

__all__ = ["JobStatus", "JobError", "Job"]


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobError(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None


class Job(BaseModel):
    """One normalized job shape across parse and extract (envelopes diverge upstream)."""

    job_id: str
    status: JobStatus
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = None
    # Populated on completion: V2ParseResponse for parse jobs, V2ExtractResult for extract jobs.
    result: Optional[object] = None
    error: Optional[JobError] = None
    # Full original envelope for fields not surfaced above (org_id, output_url, version, ...).
    raw: Dict[str, object] = {}

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
```

```python
# src/landingai_ade/types/v2/parse_response.py
from __future__ import annotations

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["V2ParseBilling", "V2ParseMetadata", "V2ParseResponse"]


class V2ParseBilling(BaseModel):
    service_tier: Optional[str] = None
    total_credits: Optional[float] = None


class V2ParseMetadata(BaseModel):
    req_id: Optional[str] = None
    job_id: Optional[str] = None
    model_version: Optional[str] = None
    page_count: Optional[int] = None
    markdown_chars: Optional[int] = None
    failed_pages: Optional[List[int]] = None
    duration_ms: Optional[int] = None
    billing: Optional[V2ParseBilling] = None


class V2ParseResponse(BaseModel):
    """V2 parse result. The gateway spec types this loosely; fields are permissive
    and extra keys are retained. Re-verify against the typed schema when the gateway
    publishes one."""

    markdown: Optional[str] = None
    structure: Optional[object] = None
    grounding: Optional[object] = None
    metadata: Optional[V2ParseMetadata] = None
```

```python
# src/landingai_ade/types/v2/extract_response.py
from __future__ import annotations

from typing import Dict, Optional

from ..._models import BaseModel

__all__ = ["V2ExtractMetadata", "V2ExtractResult"]


class V2ExtractMetadata(BaseModel):
    job_id: str
    version: str
    duration_ms: int
    doc_id: Optional[str] = None
    credit_usage: float = 0.0


class V2ExtractResult(BaseModel):
    extraction: Dict[str, object]
    extraction_metadata: Dict[str, object]
    markdown: str
    metadata: V2ExtractMetadata
```

```python
# src/landingai_ade/types/v2/file_upload_response.py
from __future__ import annotations

from typing import Optional

from ..._models import BaseModel

__all__ = ["V2FileUploadResponse"]


class V2FileUploadResponse(BaseModel):
    """`POST /v1/files` returns an open string map; `file_ref` is the key we consume."""

    file_ref: Optional[str] = None
```

```python
# src/landingai_ade/types/v2/__init__.py
from __future__ import annotations

from .job import Job as Job, JobError as JobError, JobStatus as JobStatus
from .parse_response import (
    V2ParseBilling as V2ParseBilling,
    V2ParseMetadata as V2ParseMetadata,
    V2ParseResponse as V2ParseResponse,
)
from .extract_response import (
    V2ExtractResult as V2ExtractResult,
    V2ExtractMetadata as V2ExtractMetadata,
)
from .file_upload_response import V2FileUploadResponse as V2FileUploadResponse
```

> **Note on `BaseModel` extra fields:** the repo's `_models.BaseModel` retains unknown fields, so permissive parsing works without per-model config. Verify with the `test_parse_response_tolerates_loose_shape` test (it passes `structure` as a list and nested billing).

- [ ] **Step 4: Run to verify pass**

Run: `rye run pytest tests/test_v2_types.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landingai_ade/types/v2 tests/test_v2_types.py
git commit -m "feat(v2): add V2 response and unified Job types"
```

---

## Task 3: Job normalizers

**Files:**
- Create: `src/landingai_ade/resources/v2/__init__.py` (empty for now, `# v2 resources package`), `src/landingai_ade/resources/v2/_normalize.py`
- Test: `tests/test_v2_normalize.py`

**Interfaces:**
- Consumes: `Job`, `JobStatus`, `JobError` (Task 2); `V2ParseResponse`, `V2ExtractResult` (Task 2); `parse_datetime` from `..._utils`.
- Produces:
  - `normalize_parse_job(raw: Mapping[str, object]) -> Job`
  - `normalize_extract_job(raw: Mapping[str, object]) -> Job`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v2_normalize.py
from __future__ import annotations

from landingai_ade.types.v2 import JobStatus, V2ExtractResult, V2ParseResponse
from landingai_ade.resources.v2._normalize import normalize_parse_job, normalize_extract_job


def test_normalize_parse_job_epoch_and_data() -> None:
    raw = {
        "job_id": "p1",
        "status": "completed",
        "received_at": 1_700_000_000,
        "created_at": 1_700_000_005,
        "progress": 1.0,
        "org_id": "o1",
        "output_url": None,
        "data": {"markdown": "# hi", "metadata": {"job_id": "p1", "page_count": 1}},
    }
    job = normalize_parse_job(raw)
    assert job.job_id == "p1"
    assert job.status is JobStatus.COMPLETED
    assert job.created_at is not None and job.created_at.year == 2023
    assert isinstance(job.result, V2ParseResponse)
    assert job.result.markdown == "# hi"
    assert job.error is None
    assert job.raw["org_id"] == "o1"  # envelope-only fields preserved


def test_normalize_parse_job_failure_reason() -> None:
    raw = {"job_id": "p2", "status": "failed", "failure_reason": "bad pdf", "created_at": 1_700_000_000}
    job = normalize_parse_job(raw)
    assert job.status is JobStatus.FAILED
    assert job.error is not None and job.error.message == "bad pdf"
    assert job.result is None


def test_normalize_extract_job_iso_and_result() -> None:
    raw = {
        "job_id": "e1",
        "status": "completed",
        "created_at": "2026-01-02T03:04:05Z",
        "completed_at": "2026-01-02T03:04:09Z",
        "result": {
            "extraction": {"revenue": "1M"},
            "extraction_metadata": {"revenue": {"value": "1M", "spans": []}},
            "markdown": "# doc",
            "metadata": {"job_id": "e1", "version": "extract-1", "duration_ms": 10},
        },
    }
    job = normalize_extract_job(raw)
    assert job.status is JobStatus.COMPLETED
    assert job.created_at is not None and job.created_at.year == 2026
    assert job.completed_at is not None
    assert isinstance(job.result, V2ExtractResult)
    assert job.result.metadata.version == "extract-1"


def test_normalize_extract_job_error_object() -> None:
    raw = {"job_id": "e2", "status": "failed", "error": {"code": "internal_error", "message": "boom"}}
    job = normalize_extract_job(raw)
    assert job.status is JobStatus.FAILED
    assert job.error is not None and job.error.code == "internal_error"
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/test_v2_normalize.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement normalizers**

```python
# src/landingai_ade/resources/v2/_normalize.py
from __future__ import annotations

from typing import Any, Mapping, Optional
from datetime import datetime

from ..._utils import parse_datetime
from ...types.v2 import Job, JobError, JobStatus, V2ExtractResult, V2ParseResponse

__all__ = ["normalize_parse_job", "normalize_extract_job"]


def _ts(value: object) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return parse_datetime(value)  # handles both epoch ints and ISO strings
    except (ValueError, TypeError):
        return None


def _progress(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_parse_job(raw: Mapping[str, Any]) -> Job:
    status = JobStatus(raw["status"])
    data = raw.get("data")
    result = V2ParseResponse(**data) if isinstance(data, Mapping) else None

    error = None
    reason = raw.get("failure_reason")
    if reason:
        error = JobError(message=str(reason))

    return Job(
        job_id=str(raw["job_id"]),
        status=status,
        created_at=_ts(raw.get("created_at") or raw.get("received_at")),
        completed_at=None,  # parse envelope has no completed_at
        progress=_progress(raw.get("progress")),
        result=result,
        error=error,
        raw=dict(raw),
    )


def normalize_extract_job(raw: Mapping[str, Any]) -> Job:
    status = JobStatus(raw["status"])
    payload = raw.get("result")
    result = V2ExtractResult(**payload) if isinstance(payload, Mapping) else None

    error = None
    err = raw.get("error")
    if isinstance(err, Mapping):
        error = JobError(code=err.get("code"), message=err.get("message"))
    elif raw.get("failure_reason"):  # extract *list* uses failure_reason
        error = JobError(message=str(raw["failure_reason"]))

    return Job(
        job_id=str(raw["job_id"]),
        status=status,
        created_at=_ts(raw.get("created_at")),
        completed_at=_ts(raw.get("completed_at")),
        progress=_progress(raw.get("progress")),
        result=result,
        error=error,
        raw=dict(raw),
    )
```

Also create the package init:

```python
# src/landingai_ade/resources/v2/__init__.py
# v2 resources package
```

- [ ] **Step 4: Run to verify pass**

Run: `rye run pytest tests/test_v2_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landingai_ade/resources/v2/__init__.py src/landingai_ade/resources/v2/_normalize.py tests/test_v2_normalize.py
git commit -m "feat(v2): normalize parse/extract job envelopes into one Job shape"
```

---

## Task 4: Schema coercion (pydantic / dict / str → dict)

**Files:**
- Modify: `src/landingai_ade/lib/schema_utils.py`, `src/landingai_ade/lib/__init__.py`
- Test: `tests/test_v2_schema.py`

**Interfaces:**
- Consumes: existing `_resolve_refs` in `schema_utils.py`.
- Produces: `coerce_schema_to_dict(schema: Union[str, Mapping[str, Any], Type[BaseModel]]) -> Dict[str, Any]` — returns a plain JSON-Schema dict suitable for the V2 extract JSON body.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v2_schema.py
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from landingai_ade.lib.schema_utils import coerce_schema_to_dict


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


def test_coerce_from_pydantic_model() -> None:
    out = coerce_schema_to_dict(Invoice)
    assert out["type"] == "object"
    assert "revenue" in out["properties"]
    assert "$defs" not in out  # refs resolved & stripped


def test_coerce_from_dict_passthrough() -> None:
    d = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert coerce_schema_to_dict(d) == d


def test_coerce_from_json_string() -> None:
    out = coerce_schema_to_dict('{"type": "object", "properties": {}}')
    assert out == {"type": "object", "properties": {}}


def test_coerce_rejects_garbage() -> None:
    with pytest.raises((ValueError, TypeError)):
        coerce_schema_to_dict("not json")
    with pytest.raises(TypeError):
        coerce_schema_to_dict(123)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/test_v2_schema.py -v`
Expected: FAIL (`ImportError: cannot import name 'coerce_schema_to_dict'`).

- [ ] **Step 3: Implement**

Append to `src/landingai_ade/lib/schema_utils.py`:

```python
from typing import Union, Mapping  # add alongside existing typing imports


def pydantic_to_schema_dict(model: Type[BaseModel]) -> Dict[str, Any]:
    """Like `pydantic_to_json_schema` but returns a dict with $refs resolved."""
    if not hasattr(model, "model_json_schema"):
        raise TypeError("model must be a Pydantic BaseModel subclass")
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    return _resolve_refs(schema, defs)


def coerce_schema_to_dict(schema: Union[str, Mapping[str, Any], Type[BaseModel]]) -> Dict[str, Any]:
    """Accept a pydantic model class, a dict, or a JSON string; return a JSON-Schema dict.

    The V2 extract endpoint takes `schema` as a JSON object in the request body.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return pydantic_to_schema_dict(schema)
    if isinstance(schema, Mapping):
        return dict(schema)
    if isinstance(schema, str):
        parsed = json.loads(schema)  # raises ValueError on bad JSON
        if not isinstance(parsed, dict):
            raise TypeError("schema JSON string must decode to an object")
        return parsed
    raise TypeError(f"Unsupported schema type: {type(schema)!r}")
```

Export it in `src/landingai_ade/lib/__init__.py`:

```python
from .schema_utils import pydantic_to_json_schema, coerce_schema_to_dict

__all__ = [
    "pydantic_to_json_schema",
    "coerce_schema_to_dict",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `rye run pytest tests/test_v2_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landingai_ade/lib/schema_utils.py src/landingai_ade/lib/__init__.py tests/test_v2_schema.py
git commit -m "feat(v2): accept pydantic model / dict / json string as extract schema"
```

---

## Task 5: V2 errors + sync-timeout wrapping

**Files:**
- Create: `src/landingai_ade/lib/v2_errors.py`
- Test: `tests/test_v2_normalize.py` (append a small class) — or a new `tests/test_v2_errors.py`

**Interfaces:**
- Produces:
  - `V2SyncTimeoutError(LandingAiadeError)` — raised when a sync `/v2/parse` or `/v2/extract` returns 504.
  - `JobWaitTimeoutError(LandingAiadeError)` — raised by `wait()` when the local timeout elapses.
  - `JobFailedError(LandingAiadeError)` — raised by `wait(..., raise_on_failure=True)` when a job ends `failed`/`cancelled`.
  - `raise_if_sync_timeout(exc: APIStatusError) -> None` — re-raise a 504 `APIStatusError` as `V2SyncTimeoutError`; return otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_errors.py
from __future__ import annotations

import httpx
import pytest

from landingai_ade._exceptions import APIStatusError
from landingai_ade.lib.v2_errors import (
    V2SyncTimeoutError,
    JobFailedError,
    JobWaitTimeoutError,
    raise_if_sync_timeout,
)


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://aide.landing.ai/v2/extract")
    response = httpx.Response(code, request=request)
    return APIStatusError("err", response=response, body=None)


def test_raise_if_sync_timeout_converts_504() -> None:
    with pytest.raises(V2SyncTimeoutError):
        raise_if_sync_timeout(_status_error(504))


def test_raise_if_sync_timeout_ignores_other_codes() -> None:
    raise_if_sync_timeout(_status_error(500))  # returns without raising


def test_error_hierarchy() -> None:
    from landingai_ade._exceptions import LandingAiadeError

    assert issubclass(V2SyncTimeoutError, LandingAiadeError)
    assert issubclass(JobWaitTimeoutError, LandingAiadeError)
    assert issubclass(JobFailedError, LandingAiadeError)
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/test_v2_errors.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/landingai_ade/lib/v2_errors.py
from __future__ import annotations

from .._exceptions import APIStatusError, LandingAiadeError

__all__ = [
    "V2SyncTimeoutError",
    "JobWaitTimeoutError",
    "JobFailedError",
    "raise_if_sync_timeout",
]


class V2SyncTimeoutError(LandingAiadeError):
    """A synchronous /v2/parse or /v2/extract call exceeded the server wait window (504).

    The server cancels the work; use the async jobs route (`*_jobs.create` + `wait`)
    for long-running documents."""


class JobWaitTimeoutError(LandingAiadeError):
    """`wait()` gave up before the job reached a terminal state."""


class JobFailedError(LandingAiadeError):
    """A job reached a terminal `failed`/`cancelled` state (raise_on_failure=True)."""


def raise_if_sync_timeout(exc: APIStatusError) -> None:
    if exc.response.status_code == 504:
        raise V2SyncTimeoutError(
            "The synchronous request timed out (HTTP 504). The server cancels the work on "
            "timeout — use the async jobs route (`.jobs.create(...)` then `.jobs.wait(...)`) "
            "for long-running documents."
        ) from exc
```

- [ ] **Step 4: Run to verify pass**

Run: `rye run pytest tests/test_v2_errors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landingai_ade/lib/v2_errors.py tests/test_v2_errors.py
git commit -m "feat(v2): add V2 error types and 504 sync-timeout wrapper"
```

---

## Task 6: V2 base mixin, waiter, and `client.v2` wiring

**Files:**
- Create: `src/landingai_ade/resources/v2/_base.py`, `src/landingai_ade/resources/v2/v2.py`
- Modify: `src/landingai_ade/resources/v2/__init__.py`, `src/landingai_ade/_client.py` (add `v2` cached properties)
- Test: `tests/api_resources/v2/__init__.py` (empty), `tests/api_resources/v2/test_files.py` uses this wiring in Task 7; add a focused routing test here in `tests/test_v2_environment.py`.

**Interfaces:**
- Consumes: `SyncAPIResource`/`AsyncAPIResource`, `client._v2_base_url` (Task 1).
- Produces:
  - `V2ResourceMixin` with `self._v2_url(path: str) -> str` building `f"{self._client._v2_base_url}{path}"`.
  - `poll_until_terminal(...)` sync + `apoll_until_terminal(...)` async helpers implementing backoff on a `get`-callable that returns a `Job`.
  - `V2Resource(SyncAPIResource)` exposing `.parse`, `.parse_jobs`, `.extract`, `.extract_jobs`, `.files` (cached properties); `AsyncV2Resource` mirror.
  - `LandingAIADE.v2 -> V2Resource`, `AsyncLandingAIADE.v2 -> AsyncV2Resource`.

- [ ] **Step 1: Write the failing test (routing + attachment)**

```python
# add to tests/test_v2_environment.py
import respx
import httpx

from landingai_ade import LandingAIADE

APIKEY = "My Apikey"


@respx.mock
def test_v2_subclient_routes_to_v2_host() -> None:
    c = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://aide.landing.ai/v1/files").mock(
        return_value=httpx.Response(200, json={"file_ref": "ref-123"})
    )
    ref = c.v2.files.upload(file=b"hello")
    assert route.called
    assert ref == "ref-123"


def test_v2_attribute_exists() -> None:
    c = LandingAIADE(apikey=APIKEY)
    assert c.v2 is not None
    assert c.v2.files is not None
```

> `test_v2_subclient_routes_to_v2_host` will pass only after Task 7 implements `files.upload`; keep it here to prove routing. If executing strictly task-by-task, mark it `@pytest.mark.skip` until Task 7, then unskip. `test_v2_attribute_exists` must pass at the end of this task.

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/test_v2_environment.py::test_v2_attribute_exists -v`
Expected: FAIL (`AttributeError: 'LandingAIADE' object has no attribute 'v2'`).

- [ ] **Step 3: Implement the mixin + waiter**

```python
# src/landingai_ade/resources/v2/_base.py
from __future__ import annotations

import time
from typing import Any, Callable, Optional, Awaitable, TYPE_CHECKING

import anyio

from ...types.v2 import Job
from ...lib.v2_errors import JobFailedError, JobWaitTimeoutError

if TYPE_CHECKING:
    from ..._client import LandingAIADE, AsyncLandingAIADE

DEFAULT_POLL_INITIAL = 1.0
DEFAULT_POLL_MAX = 10.0
DEFAULT_POLL_FACTOR = 1.5
DEFAULT_WAIT_TIMEOUT = 600.0


class V2ResourceMixin:
    _client: Any  # LandingAIADE | AsyncLandingAIADE

    def _v2_url(self, path: str) -> str:
        return f"{self._client._v2_base_url}{path}"


def _next_delay(current: float, poll_interval: Optional[float]) -> float:
    if poll_interval is not None:
        return poll_interval
    return min(current * DEFAULT_POLL_FACTOR, DEFAULT_POLL_MAX)


def poll_until_terminal(
    get_job: Callable[[], Job],
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    timeout: float,
    poll_interval: Optional[float],
    raise_on_failure: bool,
) -> Job:
    deadline = monotonic() + timeout
    delay = poll_interval if poll_interval is not None else DEFAULT_POLL_INITIAL
    while True:
        job = get_job()
        if job.is_terminal:
            if raise_on_failure and job.error is not None:
                raise JobFailedError(
                    f"Job {job.job_id} ended {job.status.value}: "
                    f"{job.error.message or job.error.code or 'unknown error'}"
                )
            return job
        if monotonic() >= deadline:
            raise JobWaitTimeoutError(
                f"Job {job.job_id} did not finish within {timeout}s (last status: {job.status.value})."
            )
        sleep(min(delay, max(0.0, deadline - monotonic())))
        delay = _next_delay(delay, poll_interval)


async def apoll_until_terminal(
    get_job: Callable[[], Awaitable[Job]],
    *,
    monotonic: Callable[[], float],
    timeout: float,
    poll_interval: Optional[float],
    raise_on_failure: bool,
) -> Job:
    deadline = monotonic() + timeout
    delay = poll_interval if poll_interval is not None else DEFAULT_POLL_INITIAL
    while True:
        job = await get_job()
        if job.is_terminal:
            if raise_on_failure and job.error is not None:
                raise JobFailedError(
                    f"Job {job.job_id} ended {job.status.value}: "
                    f"{job.error.message or job.error.code or 'unknown error'}"
                )
            return job
        if monotonic() >= deadline:
            raise JobWaitTimeoutError(
                f"Job {job.job_id} did not finish within {timeout}s (last status: {job.status.value})."
            )
        await anyio.sleep(min(delay, max(0.0, deadline - monotonic())))
        delay = _next_delay(delay, poll_interval)
```

> **Why inject `monotonic`/`sleep`:** the waiter tests (Tasks 9, 11) drive it with a fake clock so no real time passes. `Date.now`-style nondeterminism stays out of the SDK.

- [ ] **Step 4: Implement the containers + client wiring**

```python
# src/landingai_ade/resources/v2/v2.py
from __future__ import annotations

from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .files import FilesResource, AsyncFilesResource
from .parse import (
    ParseResource,
    AsyncParseResource,
    ParseJobsResource,
    AsyncParseJobsResource,
)
from .extract import (
    ExtractResource,
    AsyncExtractResource,
    ExtractJobsResource,
    AsyncExtractJobsResource,
)

__all__ = ["V2Resource", "AsyncV2Resource"]


class V2Resource(SyncAPIResource):
    @cached_property
    def files(self) -> FilesResource:
        return FilesResource(self._client)

    @cached_property
    def parse_jobs(self) -> ParseJobsResource:
        return ParseJobsResource(self._client)

    @cached_property
    def extract_jobs(self) -> ExtractJobsResource:
        return ExtractJobsResource(self._client)

    @cached_property
    def _parse(self) -> ParseResource:
        return ParseResource(self._client)

    @cached_property
    def _extract(self) -> ExtractResource:
        return ExtractResource(self._client)

    # Flat, verb-style entrypoints matching the design doc (client.v2.parse(...), client.v2.extract(...)).
    def parse(self, **kwargs: object):  # signature defined by ParseResource.run
        return self._parse.run(**kwargs)  # type: ignore[arg-type]

    def extract(self, **kwargs: object):
        return self._extract.run(**kwargs)  # type: ignore[arg-type]


class AsyncV2Resource(AsyncAPIResource):
    @cached_property
    def files(self) -> AsyncFilesResource:
        return AsyncFilesResource(self._client)

    @cached_property
    def parse_jobs(self) -> AsyncParseJobsResource:
        return AsyncParseJobsResource(self._client)

    @cached_property
    def extract_jobs(self) -> AsyncExtractJobsResource:
        return AsyncExtractJobsResource(self._client)

    @cached_property
    def _parse(self) -> AsyncParseResource:
        return AsyncParseResource(self._client)

    @cached_property
    def _extract(self) -> AsyncExtractResource:
        return AsyncExtractResource(self._client)

    async def parse(self, **kwargs: object):
        return await self._parse.run(**kwargs)  # type: ignore[arg-type]

    async def extract(self, **kwargs: object):
        return await self._extract.run(**kwargs)  # type: ignore[arg-type]
```

> **Design note:** `client.v2.parse(...)` and `client.v2.extract(...)` are thin delegators to `ParseResource.run` / `ExtractResource.run` (Tasks 8, 10). The explicit `run` method name keeps the sync verb (`parse`) and the jobs sub-resource (`parse_jobs`) cleanly separated, and leaves room for `client.v2.workflow` later without collision. The concrete typed signatures live on `run` — see those tasks. When implementing Task 8/10, replace the `**kwargs` delegators with explicit keyword signatures copied from `run` for full typing (do not leave `**kwargs` in the final code).

```python
# src/landingai_ade/resources/v2/__init__.py
# v2 resources package
from .v2 import V2Resource as V2Resource, AsyncV2Resource as AsyncV2Resource
```

In `src/landingai_ade/_client.py`, add a `v2` cached property to `LandingAIADE` (next to `parse_jobs`):

```python
    @cached_property
    def v2(self) -> V2Resource:
        from .resources.v2 import V2Resource

        return V2Resource(self)
```

And to `AsyncLandingAIADE`:

```python
    @cached_property
    def v2(self) -> AsyncV2Resource:
        from .resources.v2 import AsyncV2Resource

        return AsyncV2Resource(self)
```

Add the `TYPE_CHECKING` imports near the existing `parse_jobs` type import block in `_client.py`:

```python
if TYPE_CHECKING:
    from .resources import parse_jobs
    from .resources.parse_jobs import ParseJobsResource, AsyncParseJobsResource
    from .resources.v2 import V2Resource, AsyncV2Resource
```

- [ ] **Step 5: Run to verify pass**

Run: `rye run pytest tests/test_v2_environment.py::test_v2_attribute_exists -v`
Expected: PASS. (The routing test stays skipped until Task 7.)

- [ ] **Step 6: Commit**

```bash
git add src/landingai_ade/resources/v2 src/landingai_ade/_client.py tests/test_v2_environment.py
git commit -m "feat(v2): add v2 sub-client container, absolute-URL mixin, and job waiter"
```

---

## Task 7: `client.v2.files.upload`

**Files:**
- Create: `src/landingai_ade/resources/v2/files.py`, `tests/api_resources/v2/__init__.py` (empty), `tests/api_resources/v2/test_files.py`

**Interfaces:**
- Consumes: `V2ResourceMixin` (Task 6), `V2FileUploadResponse` (Task 2), `extract_files`, `deepcopy_with_paths`, `make_request_options`.
- Produces:
  - `FilesResource.upload(self, *, file: FileTypes, extra_headers=..., ..., timeout=...) -> str` (returns `file_ref`).
  - `AsyncFilesResource.upload(...) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api_resources/v2/test_files.py
from __future__ import annotations

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE, AsyncLandingAIADE

APIKEY = "My Apikey"


@respx.mock
def test_files_upload_returns_ref_and_hits_v2_host() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://aide.landing.ai/v1/files").mock(
        return_value=httpx.Response(200, json={"file_ref": "fr_1"})
    )
    ref = client.v2.files.upload(file=b"markdown bytes")
    assert route.called
    assert route.calls.last.request.headers["authorization"] == "Bearer My Apikey"
    assert ref == "fr_1"


@respx.mock
@pytest.mark.asyncio
async def test_async_files_upload() -> None:
    client = AsyncLandingAIADE(apikey=APIKEY, environment="staging")
    respx.post("https://aide.staging.landing.ai/v1/files").mock(
        return_value=httpx.Response(200, json={"file_ref": "fr_2"})
    )
    ref = await client.v2.files.upload(file=b"bytes")
    assert ref == "fr_2"
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/api_resources/v2/test_files.py -v`
Expected: FAIL (`ModuleNotFoundError` / no `files.py`).

- [ ] **Step 3: Implement**

```python
# src/landingai_ade/resources/v2/files.py
from __future__ import annotations

from typing import Mapping, cast

import httpx

from ..._files import deepcopy_with_paths
from ..._types import Body, Query, Headers, NotGiven, FileTypes, not_given
from ..._utils import extract_files
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._base_client import make_request_options
from ...types.v2 import V2FileUploadResponse
from ._base import V2ResourceMixin

__all__ = ["FilesResource", "AsyncFilesResource"]


class FilesResource(V2ResourceMixin, SyncAPIResource):
    def upload(
        self,
        *,
        file: FileTypes,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> str:
        """Stage bytes on the data plane; returns a `file_ref` for use as extract `markdown_ref`."""
        body = deepcopy_with_paths({"file": file}, [["file"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["file"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        response = self._post(
            self._v2_url("/v1/files"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2FileUploadResponse,
        )
        if not response.file_ref:
            from ...lib.v2_errors import LandingAiadeError

            raise LandingAiadeError(f"POST /v1/files did not return a file_ref (got: {response!r}).")
        return response.file_ref


class AsyncFilesResource(V2ResourceMixin, AsyncAPIResource):
    async def upload(
        self,
        *,
        file: FileTypes,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> str:
        body = deepcopy_with_paths({"file": file}, [["file"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["file"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        response = await self._post(
            self._v2_url("/v1/files"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2FileUploadResponse,
        )
        if not response.file_ref:
            from ...lib.v2_errors import LandingAiadeError

            raise LandingAiadeError(f"POST /v1/files did not return a file_ref (got: {response!r}).")
        return response.file_ref
```

> `LandingAiadeError` is re-exported from `lib.v2_errors` for the guard; if that import reads awkwardly, import it directly from `..._exceptions`.

- [ ] **Step 4: Run to verify pass, then unskip the Task 6 routing test**

Run: `rye run pytest tests/api_resources/v2/test_files.py -v`
Expected: PASS.

Remove the `@pytest.mark.skip` from `test_v2_subclient_routes_to_v2_host` in `tests/test_v2_environment.py` and run:
Run: `rye run pytest tests/test_v2_environment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landingai_ade/resources/v2/files.py tests/api_resources/v2 tests/test_v2_environment.py
git commit -m "feat(v2): add client.v2.files.upload staging on the ADE host"
```

---

## Task 8: `client.v2.parse` (sync parse) + 206 + save_to

**Files:**
- Create: `src/landingai_ade/resources/v2/parse.py` (ParseResource / AsyncParseResource — jobs classes added in Task 9)
- Modify: `src/landingai_ade/resources/v2/v2.py` (replace the `parse` `**kwargs` delegator with an explicit signature)
- Test: `tests/api_resources/v2/test_parse.py`

**Interfaces:**
- Consumes: `V2ResourceMixin`, `raise_if_sync_timeout` (Task 5), `V2ParseResponse` (Task 2), `_save_response`/`_get_input_filename` from `_client.py`, `convert_url_to_file_if_local` from `lib.url_utils`.
- Produces:
  - `ParseResource.run(self, *, document: Optional[FileTypes]=..., document_url: Optional[str]=..., model: Optional[str]=..., options: Optional[Mapping[str,object]]=..., password: Optional[str]=..., save_to: str|Path|None=None, extra_*..., timeout=...) -> V2ParseResponse`
  - `AsyncParseResource.run(...) -> V2ParseResponse`

- [ ] **Step 1: Write the failing tests**

```python
# tests/api_resources/v2/test_parse.py
from __future__ import annotations

import json

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2ParseResponse
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
PARSE_BODY = {
    "markdown": "# Hello",
    "structure": [{"type": "text"}],
    "metadata": {"req_id": "r1", "job_id": "j1", "model_version": "dpt-3", "page_count": 1, "failed_pages": []},
}


@respx.mock
def test_parse_sync_ok_routes_to_v2_and_sends_options_json() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://aide.landing.ai/v2/parse").mock(
        return_value=httpx.Response(200, json=PARSE_BODY)
    )
    result = client.v2.parse(document=b"pdf", model="dpt-3-latest", options={"foo": "bar"})
    assert isinstance(result, V2ParseResponse)
    assert result.markdown == "# Hello"
    # options must be sent as a JSON-encoded string form field
    sent = route.calls.last.request.content
    assert b'{"foo": "bar"}' in sent or b'"foo"' in sent


@respx.mock
def test_parse_sync_206_returns_response_with_failed_pages() -> None:
    client = LandingAIADE(apikey=APIKEY)
    body = dict(PARSE_BODY)
    body["metadata"] = {**PARSE_BODY["metadata"], "failed_pages": [3]}
    respx.post("https://aide.landing.ai/v2/parse").mock(return_value=httpx.Response(206, json=body))
    result = client.v2.parse(document=b"pdf")
    assert result.metadata is not None and result.metadata.failed_pages == [3]


@respx.mock
def test_parse_sync_504_raises_v2_sync_timeout() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://aide.landing.ai/v2/parse").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.parse(document=b"pdf")


@respx.mock
def test_parse_save_to_writes_file(tmp_path) -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    client.v2.parse(document=b"pdf", save_to=str(tmp_path))
    written = list(tmp_path.glob("*.json"))
    assert written and json.loads(written[0].read_text())["markdown"] == "# Hello"
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/api_resources/v2/test_parse.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `parse.py` (sync + async run)**

```python
# src/landingai_ade/resources/v2/parse.py
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, cast
from pathlib import Path

import httpx

from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import extract_files
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...types.v2 import V2ParseResponse
from ...lib.v2_errors import raise_if_sync_timeout
from ...lib.url_utils import convert_url_to_file_if_local
from ._base import V2ResourceMixin

__all__ = ["ParseResource", "AsyncParseResource"]


def _build_parse_body(
    document: object, document_url: object, model: object, options: object, password: object
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "document": document,
        "document_url": document_url,
        "model": model,
        "password": password,
    }
    # `options` is a JSON-encoded string form field per the contract.
    if options is not omit and options is not None:
        body["options"] = json.dumps(options) if not isinstance(options, str) else options
    return body


class ParseResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ParseResponse:
        """Parse a document synchronously (HTTP 200, or 206 with `metadata.failed_pages`
        when some pages failed). Raises `V2SyncTimeoutError` on server timeout (504)."""
        original_doc, original_url = document, document_url
        document, document_url = convert_url_to_file_if_local(document, document_url)
        body = deepcopy_with_paths(
            _build_parse_body(document, document_url, model, options, password), [["document"]]
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        try:
            result = self._post(
                self._v2_url("/v2/parse"),
                body=body,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ParseResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _get_input_filename, _save_response

            filename = _get_input_filename(original_doc, original_url)
            _save_response(save_to, filename, "parse", result)
        return result


class AsyncParseResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ParseResponse:
        original_doc, original_url = document, document_url
        document, document_url = convert_url_to_file_if_local(document, document_url)
        body = deepcopy_with_paths(
            _build_parse_body(document, document_url, model, options, password), [["document"]]
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        try:
            result = await self._post(
                self._v2_url("/v2/parse"),
                body=body,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ParseResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _get_input_filename, _save_response

            filename = _get_input_filename(original_doc, original_url)
            _save_response(save_to, filename, "parse", result)
        return result
```

- [ ] **Step 4: Replace the delegator in `v2.py` with an explicit signature**

In `V2Resource`, replace `def parse(self, **kwargs...)` with a full-signature delegator (copy the `run` keyword list; same for async):

```python
    def parse(
        self,
        *,
        document=omit,
        document_url=omit,
        model=omit,
        options=omit,
        password=omit,
        save_to=None,
        extra_headers=None,
        extra_query=None,
        extra_body=None,
        timeout=not_given,
    ):
        return self._parse.run(
            document=document, document_url=document_url, model=model, options=options,
            password=password, save_to=save_to, extra_headers=extra_headers,
            extra_query=extra_query, extra_body=extra_body, timeout=timeout,
        )
```

Add the needed imports to `v2.py`: `from ..._types import omit, not_given`. Mirror in `AsyncV2Resource.parse` with `await`.

- [ ] **Step 5: Run to verify pass**

Run: `rye run pytest tests/api_resources/v2/test_parse.py -v`
Expected: PASS (4 tests). Confirm 206 returns a value (base client treats 2xx as success — verified in `_base_client`; only >=500 retries/raises).

- [ ] **Step 6: Commit**

```bash
git add src/landingai_ade/resources/v2/parse.py src/landingai_ade/resources/v2/v2.py tests/api_resources/v2/test_parse.py
git commit -m "feat(v2): add client.v2.parse sync parse with 206 + save_to + 504 handling"
```

---

## Task 9: `client.v2.parse_jobs` (create / list / get / wait)

**Files:**
- Modify: `src/landingai_ade/resources/v2/parse.py` (add `ParseJobsResource`, `AsyncParseJobsResource`)
- Test: append to `tests/api_resources/v2/test_parse.py`

**Interfaces:**
- Consumes: `normalize_parse_job` (Task 3), `poll_until_terminal`/`apoll_until_terminal` (Task 6), `Job` (Task 2).
- Produces:
  - `ParseJobsResource.create(self, *, document=..., document_url=..., model=..., options=..., password=..., output_save_url=..., priority: Optional[Literal["standard","priority"]]=..., extra_*, timeout) -> Job`
  - `.list(self, *, page=..., page_size=..., status=..., ...) -> list[Job]` (returns normalized jobs; envelope `has_more`/`org_id` accessible via each job's `.raw` and via the returned object — see note)
  - `.get(self, job_id: str, ...) -> Job`
  - `.wait(self, job_id: str, *, timeout: float = 600, poll_interval: Optional[float] = None, raise_on_failure: bool = False) -> Job`
  - Async mirror.

> **List return shape:** to preserve `has_more`/`org_id` without a second type, `.list` returns a `JobList` — a `list[Job]` subclass carrying `.has_more: bool` and `.org_id: Optional[str]`. Define it in `_base.py`.

- [ ] **Step 1: Add `JobList` to `_base.py`**

```python
# append to src/landingai_ade/resources/v2/_base.py
from typing import List, Optional


class JobList(List[Job]):
    """A list of normalized jobs plus the pagination envelope."""

    has_more: bool = False
    org_id: Optional[str] = None
    page: Optional[int] = None
    page_size: Optional[int] = None

    @classmethod
    def build(cls, jobs: List[Job], **envelope: object) -> "JobList":
        out = cls(jobs)
        out.has_more = bool(envelope.get("has_more", False))
        out.org_id = cast_opt_str(envelope.get("org_id"))
        page = envelope.get("page")
        page_size = envelope.get("page_size")
        out.page = int(page) if isinstance(page, int) else None
        out.page_size = int(page_size) if isinstance(page_size, int) else None
        return out


def cast_opt_str(v: object) -> Optional[str]:
    return str(v) if isinstance(v, str) else None
```

- [ ] **Step 2: Write the failing tests (append to `test_parse.py`)**

```python
from landingai_ade.types.v2 import Job, JobStatus


@respx.mock
def test_parse_job_create_normalizes_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "p1", "status": "pending", "received_at": 1700000000})
    )
    job = client.v2.parse_jobs.create(document=b"pdf", priority="priority")
    assert isinstance(job, Job)
    assert job.job_id == "p1" and job.status is JobStatus.PENDING


@respx.mock
def test_parse_job_get_completed_has_typed_result() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://aide.landing.ai/v2/parse/jobs/p1").mock(
        return_value=httpx.Response(
            200,
            json={"job_id": "p1", "status": "completed", "created_at": 1700000000, "data": PARSE_BODY},
        )
    )
    job = client.v2.parse_jobs.get("p1")
    assert job.status is JobStatus.COMPLETED
    assert job.result is not None and job.result.markdown == "# Hello"


@respx.mock
def test_parse_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://aide.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(
            200,
            json={"jobs": [{"job_id": "p1", "status": "completed"}], "org_id": "o1", "has_more": True},
        )
    )
    jobs = client.v2.parse_jobs.list(page=0)
    assert len(jobs) == 1 and jobs[0].job_id == "p1"
    assert jobs.has_more is True and jobs.org_id == "o1"


@respx.mock
def test_parse_job_wait_polls_until_completed() -> None:
    client = LandingAIADE(apikey=APIKEY)
    responses = [
        httpx.Response(200, json={"job_id": "p1", "status": "processing", "progress": 0.5}),
        httpx.Response(200, json={"job_id": "p1", "status": "completed", "data": PARSE_BODY}),
    ]
    respx.get("https://aide.landing.ai/v2/parse/jobs/p1").mock(side_effect=responses)
    # inject fake clock/sleep so no real time passes
    from landingai_ade.resources.v2 import _base

    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    job = client.v2.parse_jobs.wait("p1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks))
    assert job.status is JobStatus.COMPLETED
```

> The `_monotonic` kwarg is a test seam; default to `time.monotonic`.

- [ ] **Step 3: Run to verify failure**

Run: `rye run pytest tests/api_resources/v2/test_parse.py -k "job" -v`
Expected: FAIL.

- [ ] **Step 4: Implement the jobs classes (append to `parse.py`)**

```python
# imports to add at top of parse.py
import time
from typing import List
from typing_extensions import Literal

from ..._utils import maybe_transform  # if using param TypedDicts; else build query dict inline
from ..._base_client import make_request_options
from ...types.v2 import Job
from ._base import JobList, poll_until_terminal, apoll_until_terminal
from ._normalize import normalize_parse_job


class ParseJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        priority: Optional[Literal["standard", "priority"]] | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        body = _build_parse_body(document, document_url, model, options, password)
        body["output_save_url"] = output_save_url
        body["priority"] = priority
        body = deepcopy_with_paths(body, [["document"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        raw = self._post(
            self._v2_url("/v2/parse/jobs"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),  # parse raw dict; see note
        )
        return normalize_parse_job(cast(Mapping[str, Any], raw))

    def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v2/parse/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_parse_job(cast(Mapping[str, Any], raw))

    def list(
        self,
        *,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        status: Optional[str] | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> JobList:
        query = {k: v for k, v in {"page": page, "page_size": page_size, "status": status}.items() if v is not omit}
        raw = self._get(
            self._v2_url("/v2/parse/jobs"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
                timeout=timeout, query=query,
            ),
            cast_to=cast("type[Any]", object),
        )
        env = cast(Mapping[str, Any], raw)
        jobs = [normalize_parse_job(j) for j in env.get("jobs", [])]
        return JobList.build(jobs, has_more=env.get("has_more"), org_id=env.get("org_id"))

    def wait(
        self,
        job_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: Optional[float] = None,
        raise_on_failure: bool = False,
        _monotonic: Optional[Any] = None,
    ) -> Job:
        return poll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            sleep=self._sleep,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )
```

> **`cast_to=object` note:** the jobs endpoints return loosely-typed envelopes that we normalize ourselves. Passing `object` (or the SDK's raw-dict pattern) makes the transport hand back the parsed JSON dict without imposing a model; `normalize_*` builds the `Job`. If `cast_to=object` does not yield a plain dict in this transport, use `cast_to=cast("type[Any]", Dict[str, object])` — verify with `test_parse_job_get_completed_has_typed_result`, adjusting only the `cast_to` expression.

Implement `AsyncParseJobsResource` identically with `async def`, `await self._post/_get`, and `wait` calling `apoll_until_terminal(lambda: self.get(job_id), ...)` (no `sleep` arg — the async waiter uses `anyio.sleep`).

- [ ] **Step 5: Run to verify pass**

Run: `rye run pytest tests/api_resources/v2/test_parse.py -v`
Expected: PASS (all parse tests).

- [ ] **Step 6: Commit**

```bash
git add src/landingai_ade/resources/v2/parse.py src/landingai_ade/resources/v2/_base.py tests/api_resources/v2/test_parse.py
git commit -m "feat(v2): add client.v2.parse_jobs create/list/get/wait with normalized Job"
```

---

## Task 10: `client.v2.extract` (sync extract) — schema coercion, idempotency, strict, 504

**Files:**
- Create: `src/landingai_ade/resources/v2/extract.py` (ExtractResource / AsyncExtractResource)
- Modify: `src/landingai_ade/resources/v2/v2.py` (explicit `extract` signature)
- Test: `tests/api_resources/v2/test_extract.py`

**Interfaces:**
- Consumes: `coerce_schema_to_dict` (Task 4), `raise_if_sync_timeout` (Task 5), `V2ExtractResult` (Task 2).
- Produces:
  - `ExtractResource.run(self, *, schema: Union[str, Mapping[str,object], Type[BaseModel]], markdown: Optional[str]=..., markdown_ref: Optional[str]=..., markdown_url: Optional[str]=..., model: Optional[str]=..., strict: Optional[bool]=..., idempotency_key: Optional[str]=..., save_to=None, extra_*, timeout) -> V2ExtractResult` (JSON body).
  - Async mirror.

- [ ] **Step 1: Write the failing tests**

```python
# tests/api_resources/v2/test_extract.py
from __future__ import annotations

import json

import httpx
import respx
import pytest
from pydantic import BaseModel, Field

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2ExtractResult
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
EXTRACT_BODY = {
    "extraction": {"revenue": "1M"},
    "extraction_metadata": {"revenue": {"value": "1M", "spans": []}},
    "markdown": "# doc",
    "metadata": {"job_id": "e1", "version": "extract-1", "duration_ms": 5},
}


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


@respx.mock
def test_extract_sync_json_body_with_pydantic_schema() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://aide.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    result = client.v2.extract(schema=Invoice, markdown="# doc", idempotency_key="k1")
    assert isinstance(result, V2ExtractResult) and result.metadata.version == "extract-1"
    req = json.loads(route.calls.last.request.content)
    assert req["schema"]["type"] == "object" and "revenue" in req["schema"]["properties"]
    assert req["markdown"] == "# doc"
    assert req["idempotency_key"] == "k1"
    assert route.calls.last.request.headers["content-type"].startswith("application/json")


@respx.mock
def test_extract_sync_strict_option() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://aide.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(schema={"type": "object", "properties": {}}, markdown_url="https://x/y.md", strict=True)
    req = json.loads(route.calls.last.request.content)
    assert req["options"]["strict"] is True
    assert req["markdown_url"] == "https://x/y.md"


@respx.mock
def test_extract_sync_504() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://aide.landing.ai/v2/extract").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.extract(schema={"type": "object"}, markdown="x")
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/api_resources/v2/test_extract.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `extract.py` (run methods)**

```python
# src/landingai_ade/resources/v2/extract.py
from __future__ import annotations

from typing import Any, Dict, Type, Mapping, Optional, Union
from pathlib import Path

import httpx
from pydantic import BaseModel

from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...types.v2 import V2ExtractResult
from ...lib.v2_errors import raise_if_sync_timeout
from ...lib.schema_utils import coerce_schema_to_dict
from ._base import V2ResourceMixin

__all__ = ["ExtractResource", "AsyncExtractResource"]


def _build_extract_body(
    schema: Union[str, Mapping[str, object], Type[BaseModel]],
    markdown: object,
    markdown_ref: object,
    markdown_url: object,
    model: object,
    strict: object,
    idempotency_key: object,
    priority: object = omit,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"schema": coerce_schema_to_dict(schema)}
    for key, value in (
        ("markdown", markdown),
        ("markdown_ref", markdown_ref),
        ("markdown_url", markdown_url),
        ("model", model),
        ("idempotency_key", idempotency_key),
        ("priority", priority),
    ):
        if value is not omit and value is not None:
            body[key] = value
    if strict is not omit and strict is not None:
        body["options"] = {"strict": bool(strict)}
    return body


class ExtractResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        """Extract structured data from markdown synchronously (JSON body).
        Raises `V2SyncTimeoutError` on server timeout (504); strict-mode schema
        rejection surfaces as the transport's 422 error."""
        body = _build_extract_body(schema, markdown, markdown_ref, markdown_url, model, strict, idempotency_key)
        try:
            result = self._post(
                self._v2_url("/v2/extract"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ExtractResult,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _get_input_filename, _save_response

            filename = _get_input_filename(None, markdown_url if isinstance(markdown_url, str) else None)
            _save_response(save_to, filename, "extract", result)
        return result


class AsyncExtractResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        body = _build_extract_body(schema, markdown, markdown_ref, markdown_url, model, strict, idempotency_key)
        try:
            result = await self._post(
                self._v2_url("/v2/extract"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ExtractResult,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _get_input_filename, _save_response

            filename = _get_input_filename(None, markdown_url if isinstance(markdown_url, str) else None)
            _save_response(save_to, filename, "extract", result)
        return result
```

> **Content-Type:** no `multipart/form-data` header is set, so the transport sends `application/json` (default), which the test asserts. `schema` goes as a nested JSON object — exactly what `/v2/extract` expects.

- [ ] **Step 4: Wire the explicit `extract` delegator in `v2.py`** (mirror the parse delegator, with the extract keyword list; sync + async).

- [ ] **Step 5: Run to verify pass**

Run: `rye run pytest tests/api_resources/v2/test_extract.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/landingai_ade/resources/v2/extract.py src/landingai_ade/resources/v2/v2.py tests/api_resources/v2/test_extract.py
git commit -m "feat(v2): add client.v2.extract sync extract with schema coercion + idempotency"
```

---

## Task 11: `client.v2.extract_jobs` (create / list / get / wait)

**Files:**
- Modify: `src/landingai_ade/resources/v2/extract.py` (add `ExtractJobsResource`, `AsyncExtractJobsResource`)
- Test: append to `tests/api_resources/v2/test_extract.py`

**Interfaces:**
- Consumes: `normalize_extract_job` (Task 3), `poll_until_terminal`/`apoll_until_terminal`, `JobList`.
- Produces:
  - `ExtractJobsResource.create(self, *, schema, markdown=..., markdown_ref=..., markdown_url=..., model=..., strict=..., idempotency_key=..., priority=..., ...) -> Job`
  - `.list(...) -> JobList`, `.get(job_id) -> Job`, `.wait(job_id, ...) -> Job` (+ async).

- [ ] **Step 1: Write the failing tests (append)**

```python
from landingai_ade.types.v2 import Job, JobStatus


@respx.mock
def test_extract_job_create_and_get() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "e1", "status": "pending", "created_at": "2026-01-01T00:00:00Z"})
    )
    job = client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x", priority="priority")
    assert job.job_id == "e1" and job.status is JobStatus.PENDING

    respx.get("https://aide.landing.ai/v2/extract/jobs/e1").mock(
        return_value=httpx.Response(
            200,
            json={"job_id": "e1", "status": "completed", "created_at": "2026-01-01T00:00:00Z",
                  "completed_at": "2026-01-01T00:00:09Z", "result": EXTRACT_BODY},
        )
    )
    done = client.v2.extract_jobs.get("e1")
    assert done.status is JobStatus.COMPLETED
    assert done.result is not None and done.result.metadata.version == "extract-1"


@respx.mock
def test_extract_job_get_failed_maps_error_object() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://aide.landing.ai/v2/extract/jobs/e2").mock(
        return_value=httpx.Response(
            200, json={"job_id": "e2", "status": "failed", "error": {"code": "internal_error", "message": "boom"}}
        )
    )
    job = client.v2.extract_jobs.get("e2")
    assert job.status is JobStatus.FAILED and job.error is not None and job.error.code == "internal_error"


@respx.mock
def test_extract_job_wait_raise_on_failure() -> None:
    from landingai_ade.lib.v2_errors import JobFailedError

    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://aide.landing.ai/v2/extract/jobs/e3").mock(
        return_value=httpx.Response(200, json={"job_id": "e3", "status": "failed", "error": {"code": "x", "message": "no"}})
    )
    with pytest.raises(JobFailedError):
        client.v2.extract_jobs.wait("e3", timeout=5, poll_interval=0.01, raise_on_failure=True,
                                    _monotonic=lambda: 0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `rye run pytest tests/api_resources/v2/test_extract.py -k "job" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the extract jobs classes** (append to `extract.py`) — same structure as `ParseJobsResource` (Task 9) but:
  - create uses `_build_extract_body(..., priority=priority)` as the JSON body (no multipart, no files);
  - `get`/`list` build absolute URLs `"/v2/extract/jobs"` / `f"/v2/extract/jobs/{job_id}"`;
  - normalize with `normalize_extract_job`;
  - `list` maps `page`/`page_size`/`has_more` into `JobList`.

```python
# imports to add
import time
from typing_extensions import Literal
from ...types.v2 import Job
from ._base import JobList, poll_until_terminal, apoll_until_terminal
from ._normalize import normalize_extract_job


class ExtractJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        priority: Optional[Literal["standard", "priority"]] | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        body = _build_extract_body(schema, markdown, markdown_ref, markdown_url, model, strict, idempotency_key, priority)
        raw = self._post(
            self._v2_url("/v2/extract/jobs"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_extract_job(cast(Mapping[str, Any], raw))

    def get(self, job_id: str, *, extra_headers=None, extra_query=None, extra_body=None, timeout=not_given) -> Job:
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v2/extract/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_extract_job(cast(Mapping[str, Any], raw))

    def list(self, *, page=omit, page_size=omit, status=omit, extra_headers=None, extra_query=None, extra_body=None, timeout=not_given) -> JobList:
        query = {k: v for k, v in {"page": page, "page_size": page_size, "status": status}.items() if v is not omit}
        raw = self._get(
            self._v2_url("/v2/extract/jobs"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=query
            ),
            cast_to=cast("type[Any]", object),
        )
        env = cast(Mapping[str, Any], raw)
        jobs = [normalize_extract_job(j) for j in env.get("jobs", [])]
        return JobList.build(jobs, has_more=env.get("has_more"), page=env.get("page"), page_size=env.get("page_size"))

    def wait(self, job_id: str, *, timeout: float = 600.0, poll_interval: Optional[float] = None,
             raise_on_failure: bool = False, _monotonic: Optional[Any] = None) -> Job:
        return poll_until_terminal(
            lambda: self.get(job_id), monotonic=_monotonic or time.monotonic, sleep=self._sleep,
            timeout=timeout, poll_interval=poll_interval, raise_on_failure=raise_on_failure,
        )
```

Add `cast`, `Any` to the `extract.py` typing imports. Implement `AsyncExtractJobsResource` with `async`/`await` and `apoll_until_terminal`.

- [ ] **Step 4: Run to verify pass**

Run: `rye run pytest tests/api_resources/v2/test_extract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landingai_ade/resources/v2/extract.py tests/api_resources/v2/test_extract.py
git commit -m "feat(v2): add client.v2.extract_jobs create/list/get/wait"
```

---

## Task 12: Async parity smoke test + full suite / lint / typecheck

**Files:**
- Test: `tests/api_resources/v2/test_async_smoke.py`

**Interfaces:** none new — verifies the async resources implemented alongside each sync class in Tasks 7–11.

- [ ] **Step 1: Write an async smoke test covering one route per resource**

```python
# tests/api_resources/v2/test_async_smoke.py
from __future__ import annotations

import httpx
import respx
import pytest

from landingai_ade import AsyncLandingAIADE
from landingai_ade.types.v2 import Job, JobStatus, V2ExtractResult, V2ParseResponse

APIKEY = "My Apikey"


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_and_jobs() -> None:
    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/parse").mock(
        return_value=httpx.Response(200, json={"markdown": "# a", "metadata": {"job_id": "j"}})
    )
    assert isinstance(await client.v2.parse(document=b"x"), V2ParseResponse)

    respx.post("https://aide.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "p1", "status": "pending"})
    )
    job = await client.v2.parse_jobs.create(document=b"x")
    assert isinstance(job, Job) and job.status is JobStatus.PENDING


@respx.mock
@pytest.mark.asyncio
async def test_async_extract() -> None:
    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json={
            "extraction": {}, "extraction_metadata": {}, "markdown": "m",
            "metadata": {"job_id": "e", "version": "v", "duration_ms": 1}})
    )
    r = await client.v2.extract(schema={"type": "object"}, markdown="m")
    assert isinstance(r, V2ExtractResult)
```

- [ ] **Step 2: Run the full V2 suite**

Run: `rye run pytest tests/test_v2_environment.py tests/test_v2_types.py tests/test_v2_normalize.py tests/test_v2_schema.py tests/test_v2_errors.py tests/api_resources/v2 -v`
Expected: PASS.

- [ ] **Step 3: Run the whole suite (no V1 regressions)**

Run: `rye run pytest -q`
Expected: PASS (existing skipped mock-server tests remain skipped; nothing new fails).

- [ ] **Step 4: Lint + typecheck**

Run: `rye run lint`
Expected: PASS (ruff + pyright + mypy clean). Fix any typing issues surfaced by the `cast_to=object` pattern by adjusting only the `cast_to` expressions and adding `# type: ignore` only where the transport's generics genuinely can't express the raw-dict return.

- [ ] **Step 5: Commit**

```bash
git add tests/api_resources/v2/test_async_smoke.py
git commit -m "test(v2): async parity smoke tests; full suite green"
```

---

## Task 13: Documentation & examples

**Files:**
- Modify: `api.md` (add a `## V2` section), `README.md` (add a V2 usage subsection)
- Create: `examples/v2_parse.py`, `examples/v2_extract.py`

**Interfaces:** none.

- [ ] **Step 1: Add the `api.md` V2 section**

Document, under a new `# V2` heading: `client.v2.parse`, `client.v2.parse_jobs.{create,list,get,wait}`, `client.v2.extract`, `client.v2.extract_jobs.{create,list,get,wait}`, `client.v2.files.upload`, and the `Job`, `JobStatus`, `JobError`, `V2ParseResponse`, `V2ExtractResult` types (import path `landingai_ade.types.v2`). Note the unified `Job` shape and `.raw` escape hatch.

- [ ] **Step 2: Add the README V2 subsection**

Show: environment selection (`environment="staging"` and `LANDINGAI_ADE_ENVIRONMENT`), sync parse/extract, async jobs + `wait`, pydantic-model schema, `files.upload` → `markdown_ref`, `save_to`. State explicitly: additive to V1; V1 usage unchanged.

- [ ] **Step 3: Create runnable examples**

```python
# examples/v2_parse.py
#!/usr/bin/env -S rye run python
from __future__ import annotations

from landingai_ade import LandingAIADE

client = LandingAIADE()  # apikey from VISION_AGENT_API_KEY; environment from LANDINGAI_ADE_ENVIRONMENT
job = client.v2.parse_jobs.create(document="./sample.pdf", priority="priority")
done = client.v2.parse_jobs.wait(job.job_id, timeout=600)
print(done.status, None if done.result is None else done.result.markdown[:200])
```

```python
# examples/v2_extract.py
#!/usr/bin/env -S rye run python
from __future__ import annotations

from pydantic import BaseModel, Field
from landingai_ade import LandingAIADE


class Invoice(BaseModel):
    total: str = Field(description="Invoice grand total")


client = LandingAIADE()
result = client.v2.extract(schema=Invoice, markdown_url="https://example.com/doc.md", strict=False)
print(result.extraction)
```

- [ ] **Step 4: Sanity check the examples import-compile**

Run: `rye run python -c "import ast; ast.parse(open('examples/v2_parse.py').read()); ast.parse(open('examples/v2_extract.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add api.md README.md examples/v2_parse.py examples/v2_extract.py
git commit -m "docs(v2): document V2 sub-client, environments, and add examples"
```

---

## Open items to carry back to the aide team (not blockers for this plan)

- **Parse `idempotency_key`:** absent from parse routes in the spec though present on extract. Confirm whether parse should gain it; if so, add to `ParseJobsResource.create` (one-line body field).
- **Envelope unification:** if aide unifies the parse/extract job envelopes before GA, the normalizers collapse toward identity but the public `Job` shape stays fixed — no SDK breaking change.
- **`/v2/workflow`:** out of scope here; the `V2Resource` container + `JobList`/waiter machinery accept a `workflow` / `workflow_jobs` resource later with no rework.
- **Typed parse response:** `V2ParseResponse` is permissive because the gateway types the parse result loosely; tighten when a real schema is published.
- **504 retries:** the transport retries 5xx (including 504) before `raise_if_sync_timeout` converts it. Acceptable, but consider suppressing retries specifically for sync 504s in a follow-up (they always mean "cancelled, won't complete").

---

## Self-Review

**Spec coverage:** environment matrix (Task 1) ✓; sync parse + 206 + password/options (Task 8) ✓; parse jobs + priority + output_save_url + wait (Task 9) ✓; sync extract + schema coercion + idempotency + strict (Task 10) ✓; extract jobs (Task 11) ✓; `/v1/files` on ADE host (Task 7) ✓; unified Job across divergent envelopes (Tasks 2,3) ✓; pydantic/dict/str schema (Task 4) ✓; save_to parity (Tasks 8,10) ✓; env-var QA workflow + base_url follow-through (Task 1) ✓; async parity (Tasks 7–12) ✓; docs (Task 13) ✓; `/v2/workflow` explicitly deferred with a non-blocking hook ✓.

**Type consistency:** `Job`/`JobStatus`/`JobError` names identical across Tasks 2/3/6/9/11; `normalize_parse_job`/`normalize_extract_job` signatures match their call sites; `poll_until_terminal`/`apoll_until_terminal` params match `wait` call sites; `coerce_schema_to_dict` signature matches Task 10 usage; `_v2_url`/`_v2_base_url` consistent between Task 1 (attr) and Task 6+ (consumer); `V2ResourceMixin` mixed in before `SyncAPIResource`/`AsyncAPIResource` everywhere so `self._client` is set by the resource base `__init__`.

**Placeholder scan:** no TBD/TODO; every code step carries runnable code. The two `**kwargs` delegators in Task 6's `v2.py` are explicitly replaced with typed signatures in Tasks 8 and 10 (called out in-step), so they never survive into final code.
