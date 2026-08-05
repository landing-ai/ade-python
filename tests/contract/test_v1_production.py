from __future__ import annotations

import os
import json
import time
from typing import TypeVar, Callable, Iterable, Iterator, cast
from pathlib import Path

import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types import (
    ParseResponse,
    SplitResponse,
    ExtractResponse,
    SectionResponse,
    ClassifyResponse,
    ParseJobGetResponse,
    ParseJobListResponse,
    ExtractJobGetResponse,
    ExtractJobListResponse,
    ExtractBuildSchemaResponse,
    client_split_params,
    client_classify_params,
)

# `production`, deliberately NOT `contract`: the staging gate in pr-gates.yml runs
# `pytest tests/contract -m contract`, and these tests must not ride along on every
# spec-sync PR. They hit the LIVE PRODUCTION API and spend real credits, so they run
# only from .github/workflows/e2e-production.yml (manual dispatch, plus the pre-tag
# gate in release.yml).
pytestmark = pytest.mark.production

PRODUCTION_KEY = os.environ.get("LANDINGAI_ADE_PRODUCTION_APIKEY")

# The same fixture the V2 staging smoke uses: a real 2-page PDF with text, tables and
# figures. Page count is asserted below, so swapping this file means updating that.
SAMPLE_PDF = Path(__file__).parent / "sample.pdf"
SAMPLE_PAGE_COUNT = 2

# V1 `extract` takes the JSON Schema pre-serialized as a string (unlike
# `client.v2.extract`, which also accepts a pydantic model).
EXTRACT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The document title"},
        },
    }
)

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# `classes` / `split_class` are nested arrays on a multipart/form-data body, and the V1
# spec notes for both that they "can be provided as a JSON string in form data" — that
# JSON string is the wire format the API actually parses (and what the README shows).
# Passing a real list instead lets the client form-encode it as `classes[][class]=...`,
# which the API does not accept. The generated signatures still say `Iterable[...]`, so
# cast to keep the proven wire format under pyright's strict mode.
CLASSIFY_CLASSES = cast(
    Iterable[client_classify_params.Class],
    json.dumps(
        [
            {"class": "invoice", "description": "A bill for goods or services."},
            {"class": "report", "description": "A narrative or financial report."},
        ]
    ),
)

SPLIT_CLASSES = cast(
    Iterable[client_split_params.SplitClass],
    json.dumps(
        [
            {"name": "financials", "description": "Sections presenting figures or tables."},
            {"name": "narrative", "description": "Prose sections."},
        ]
    ),
)

# Both V1 job-get responses expose `job_id` / `status` / `progress`, so one polling
# helper serves both. Constrained (not bound) so the return type stays exact.
JobT = TypeVar("JobT", ParseJobGetResponse, ExtractJobGetResponse)

# Same idea for the two list responses, which both expose `.jobs`.
ListT = TypeVar("ListT", ParseJobListResponse, ExtractJobListResponse)


@pytest.fixture(scope="module")
def production_client() -> Iterator[LandingAIADE]:
    if not PRODUCTION_KEY:
        pytest.skip("LANDINGAI_ADE_PRODUCTION_APIKEY not set")
    # `environment="production"` resolves the V1 host from ENVIRONMENTS, so this also
    # covers the environment map rather than hardcoding api.va.landing.ai.
    with LandingAIADE(apikey=PRODUCTION_KEY, environment="production") as client:
        yield client


@pytest.fixture(scope="module")
def parsed(production_client: LandingAIADE) -> ParseResponse:
    """One real parse, shared by every markdown-consuming test in this module.

    `section` specifically needs the anchor-annotated markdown that a parse emits, and
    extract / split / build-schema all work off the same text. Parsing once holds the
    per-run production spend to a single parse instead of five.
    """
    return production_client.parse(document=SAMPLE_PDF, split="page")


def _wait_for_job(
    get: Callable[[str], JobT],
    job_id: str,
    *,
    timeout: float = 300,
    poll_interval: float = 3,
) -> JobT:
    """Poll a V1 job to a terminal status.

    V1 has no `wait` helper (unlike `client.v2.parse_jobs.wait`), so the polling loop
    lives here rather than in the SDK.
    """
    deadline = time.monotonic() + timeout
    while True:
        job = get(job_id)
        if job.status in TERMINAL_STATUSES:
            return job
        if time.monotonic() >= deadline:
            raise AssertionError(f"job {job_id} still {job.status!r} after {timeout}s")
        time.sleep(poll_interval)


def _list_completed_nonempty(list_completed: Callable[[], ListT]) -> ListT:
    """Fetch a page of completed jobs, retrying briefly until it is non-empty.

    Each caller has just completed a job under this key, so the completed-jobs list must
    be non-empty; the short retry only absorbs read-after-write lag in list indexing.
    The last response is returned either way, so the caller's non-empty assertion fails
    with context if the list never populates — rather than the status-filter loop passing
    vacuously on an empty page.
    """
    listed = list_completed()
    for _ in range(4):
        if listed.jobs:
            break
        time.sleep(3)
        listed = list_completed()
    return listed


def test_parse(parsed: ParseResponse) -> None:
    """`POST /v1/ade/parse` — markdown, chunks, and per-page splits."""
    assert isinstance(parsed, ParseResponse)
    assert parsed.markdown
    assert parsed.chunks
    for chunk in parsed.chunks:
        assert chunk.id
        assert chunk.type
        assert chunk.grounding.page >= 0

    assert parsed.metadata.page_count == SAMPLE_PAGE_COUNT
    assert parsed.metadata.job_id
    # `split="page"` splits at the page level, so there is exactly one split per page,
    # each naming the chunk ids it covers.
    assert len(parsed.splits) == SAMPLE_PAGE_COUNT
    for split in parsed.splits:
        assert split.markdown
        assert split.pages
        assert split.chunks


def test_extract(production_client: LandingAIADE, parsed: ParseResponse) -> None:
    """`POST /v1/ade/extract` — schema-driven extraction off parsed markdown."""
    res = production_client.extract(schema=EXTRACT_SCHEMA, markdown=parsed.markdown)
    assert isinstance(res, ExtractResponse)
    assert isinstance(res.extraction, dict)
    assert isinstance(res.extraction_metadata, dict)
    assert res.metadata.job_id
    # Set only when the output could not be made to conform to the schema; a healthy
    # extraction against this schema leaves it null.
    assert res.metadata.schema_violation_error is None


def test_extract_build_schema(production_client: LandingAIADE, parsed: ParseResponse) -> None:
    """`POST /v1/ade/extract/build-schema` — schema generated from a document."""
    res = production_client.extract_build_schema(
        markdowns=[parsed.markdown],
        prompt="Capture the document title and any totals it reports.",
    )
    assert isinstance(res, ExtractBuildSchemaResponse)
    # `extraction_schema` is the JSON Schema serialized as a *string*, not an object.
    decoded: object = json.loads(res.extraction_schema)
    assert isinstance(decoded, dict)
    generated = cast("dict[str, object]", decoded)
    assert generated.get("type") == "object"
    assert generated.get("properties")


def test_classify(production_client: LandingAIADE) -> None:
    """`POST /v1/ade/classify` — one class assigned per page."""
    res = production_client.classify(document=SAMPLE_PDF, classes=CLASSIFY_CLASSES)
    assert isinstance(res, ClassifyResponse)
    assert res.metadata.page_count == SAMPLE_PAGE_COUNT
    # Exactly one classification per page: pages that cannot be classified come back
    # as 'unknown' rather than being dropped, so the count is stable.
    assert len(res.classification) == SAMPLE_PAGE_COUNT
    assert len({page.page for page in res.classification}) == SAMPLE_PAGE_COUNT
    for page in res.classification:
        assert page.class_


def test_section(production_client: LandingAIADE, parsed: ParseResponse) -> None:
    """`POST /v1/ade/section` — table of contents built from parsed markdown."""
    res = production_client.section(markdown=parsed.markdown)
    assert isinstance(res, SectionResponse)
    assert isinstance(res.table_of_contents_md, str)
    # Not asserted non-empty: a document with no headings legitimately sections into an
    # empty table of contents. Every entry that *is* returned must be well-formed.
    for entry in res.table_of_contents:
        assert entry.title
        assert entry.start_reference
        assert entry.level >= 0


def test_split(production_client: LandingAIADE, parsed: ParseResponse) -> None:
    """`POST /v1/ade/split` — markdown partitioned into classified groups."""
    res = production_client.split(markdown=parsed.markdown, split_class=SPLIT_CLASSES)
    assert isinstance(res, SplitResponse)
    assert res.metadata.page_count == SAMPLE_PAGE_COUNT
    assert res.splits
    for split in res.splits:
        assert split.classification
        assert split.markdowns
        assert split.pages


def test_parse_jobs(production_client: LandingAIADE) -> None:
    """`/v1/ade/parse/jobs` — create, poll to completion, and list."""
    created = production_client.parse_jobs.create(document=SAMPLE_PDF)
    assert created.job_id

    job = _wait_for_job(production_client.parse_jobs.get, created.job_id)
    assert job.status == "completed", job.failure_reason
    assert job.progress == 1
    assert job.job_id == created.job_id

    # Delivery is either inline on `data` or, for results over 1MB (and for ZDR orgs),
    # a presigned `output_url`. sample.pdf sits near that boundary once grounding is
    # included, so accept both rather than pinning one.
    assert job.data is not None or job.output_url is not None
    if job.data is not None:
        assert job.data.markdown
        assert job.data.chunks

    # Verify the `status` filter server-side. We just completed a parse job under this
    # key, so the completed list must be non-empty — assert that (with a short retry for
    # list-indexing lag) so a `list` regression that always returns `[]` can't pass this
    # gate vacuously. We assert the filter holds rather than that our specific job is on
    # page 0, which would be flaky on a busy production org.
    listed = _list_completed_nonempty(
        lambda: production_client.parse_jobs.list(page=0, page_size=10, status="completed")
    )
    assert listed.jobs, "expected at least one completed parse job (this test just completed one)"
    for listed_job in listed.jobs:
        assert listed_job.job_id
        assert listed_job.status == "completed"


def test_extract_jobs(production_client: LandingAIADE, parsed: ParseResponse) -> None:
    """`/v1/ade/extract/jobs` — create, poll to completion, and list."""
    # Unlike the sync `/extract` endpoint (test_extract), `/extract/jobs` requires
    # `markdown` as a genuine file part. The SDK treats a bare `str` as file content
    # (see _files.is_file_content) but uploads it with no filename, which this endpoint
    # rejects with 422 "Invalid file provided". Send it as a named (filename, bytes,
    # content-type) tuple so it uploads as a proper multipart file.
    markdown_file = ("document.md", parsed.markdown.encode("utf-8"), "text/markdown")
    created = production_client.extract_jobs.create(schema=EXTRACT_SCHEMA, markdown=markdown_file)
    assert created.job_id

    job = _wait_for_job(production_client.extract_jobs.get, created.job_id)
    assert job.status == "completed", job.failure_reason
    assert job.progress == 1
    assert job.job_id == created.job_id

    assert job.data is not None or job.output_url is not None
    if job.data is not None:
        assert isinstance(job.data.extraction, dict)
        assert isinstance(job.data.extraction_metadata, dict)
        assert job.data.metadata.job_id

    listed = _list_completed_nonempty(
        lambda: production_client.extract_jobs.list(page=0, page_size=10, status="completed")
    )
    assert listed.jobs, "expected at least one completed extract job (this test just completed one)"
    for listed_job in listed.jobs:
        assert listed_job.job_id
        assert listed_job.status == "completed"
