from __future__ import annotations

from typing import Any, Dict

import httpx
import respx
import pytest

from landingai_ade import AsyncLandingAIADE
from landingai_ade.types.v2 import Job, JobStatus, V2ExtractResult, V2ParseResponse

APIKEY = "My Apikey"

EXTRACT_BODY: Dict[str, Any] = {
    "extraction": {},
    "extraction_metadata": {},
    "markdown": "m",
    "metadata": {"job_id": "e1", "version": "v", "duration_ms": 1},
}


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
    respx.post("https://aide.landing.ai/v2/extract").mock(return_value=httpx.Response(200, json=EXTRACT_BODY))
    r = await client.v2.extract(schema={"type": "object"}, markdown="m")
    assert isinstance(r, V2ExtractResult)


@respx.mock
@pytest.mark.asyncio
async def test_async_extract_jobs_create_get_and_wait() -> None:
    """AsyncExtractJobsResource had zero coverage prior to this test: exercise
    create/get/wait end to end on the async client with a fake clock so no real
    time passes while polling."""
    client = AsyncLandingAIADE(apikey=APIKEY)

    respx.post("https://aide.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "e1", "status": "pending"})
    )
    created = await client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x")
    assert isinstance(created, Job)
    assert created.job_id == "e1" and created.status is JobStatus.PENDING

    responses = [
        httpx.Response(200, json={"job_id": "e1", "status": "processing", "progress": 0.5}),
        httpx.Response(200, json={"job_id": "e1", "status": "completed", "result": EXTRACT_BODY}),
    ]
    respx.get("https://aide.landing.ai/v2/extract/jobs/e1").mock(side_effect=responses)

    fetched = await client.v2.extract_jobs.get("e1")
    assert fetched.status is JobStatus.PROCESSING

    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    waited = await client.v2.extract_jobs.wait(
        "e1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks)
    )
    assert waited.status is JobStatus.COMPLETED
    assert isinstance(waited.result, V2ExtractResult)
