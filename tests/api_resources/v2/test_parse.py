from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import Job, JobStatus, V2ParseResponse
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
PARSE_BODY: Dict[str, Any] = {
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
def test_parse_sync_omits_unset_fields_from_multipart_body() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://aide.landing.ai/v2/parse").mock(
        return_value=httpx.Response(200, json=PARSE_BODY)
    )
    client.v2.parse(document=b"pdf")
    # Unset `Omit`/`NotGiven` sentinels must never leak into the multipart
    # body as literal `"<...Omit object...>"` / `"NOT_GIVEN"` form fields.
    sent = route.calls.last.request.content
    assert b"Omit object" not in sent
    assert b"NOT_GIVEN" not in sent
    assert b"document_url" not in sent
    assert b"model" not in sent
    assert b"options" not in sent
    assert b"password" not in sent


@respx.mock
def test_parse_sync_206_returns_response_with_failed_pages() -> None:
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(PARSE_BODY)
    metadata: Dict[str, Any] = dict(PARSE_BODY["metadata"])
    metadata["failed_pages"] = [3]
    body["metadata"] = metadata
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
def test_parse_save_to_writes_file(tmp_path: Path) -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    client.v2.parse(document=b"pdf", save_to=str(tmp_path))
    written = list(tmp_path.glob("*.json"))
    assert written and json.loads(written[0].read_text())["markdown"] == "# Hello"


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_sync_ok() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY, environment="production")
    respx.post("https://aide.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    result = await client.v2.parse(document=b"pdf")
    assert isinstance(result, V2ParseResponse)
    assert result.markdown == "# Hello"


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
    assert isinstance(job.result, V2ParseResponse)
    assert job.result.markdown == "# Hello"


@respx.mock
def test_parse_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.parse_jobs.get("")


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
    # inject fake clock so no real time passes
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    job = client.v2.parse_jobs.wait("p1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks))
    assert job.status is JobStatus.COMPLETED


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_job_create_and_get() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://aide.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "p1", "status": "pending"})
    )
    respx.get("https://aide.landing.ai/v2/parse/jobs/p1").mock(
        return_value=httpx.Response(200, json={"job_id": "p1", "status": "completed", "data": PARSE_BODY})
    )
    created = await client.v2.parse_jobs.create(document=b"pdf")
    assert created.status is JobStatus.PENDING
    fetched = await client.v2.parse_jobs.get("p1")
    assert fetched.status is JobStatus.COMPLETED
    assert isinstance(fetched.result, V2ParseResponse)
    assert fetched.result.markdown == "# Hello"


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_job_wait_polls_until_completed() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    responses = [
        httpx.Response(200, json={"job_id": "p1", "status": "processing", "progress": 0.5}),
        httpx.Response(200, json={"job_id": "p1", "status": "completed", "data": PARSE_BODY}),
    ]
    respx.get("https://aide.landing.ai/v2/parse/jobs/p1").mock(side_effect=responses)
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    job = await client.v2.parse_jobs.wait("p1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks))
    assert job.status is JobStatus.COMPLETED
