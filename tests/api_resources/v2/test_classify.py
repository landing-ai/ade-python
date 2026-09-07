from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import Job, JobStatus, V2ClassifyResponse
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"

CLASSES = [
    {"class": "invoice", "description": "A billing document"},
    {"class": "receipt"},
]

CLASSIFY_BODY: Dict[str, Any] = {
    "classification": [
        {"class": "invoice", "page": 0, "reason": "Has an invoice header.", "suggested_class": None},
        {"class": "unknown", "page": 1, "reason": "No clear signal.", "suggested_class": "receipt"},
    ],
    "metadata": {
        "page_count": 2,
        "duration_ms": 42,
        "openapi_spec": "https://api.ade.landing.ai/openapi.json",
        "credit_usage": 0.2,
        "filename": "doc.pdf",
        "job_id": "classify-1",
        "org_id": "o1",
        "version": "classify-20260420",
    },
}


@respx.mock
def test_classify_sync_routes_to_v2_and_encodes_classes() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v2/classify").mock(
        return_value=httpx.Response(200, json=CLASSIFY_BODY)
    )
    result = client.v2.classify(classes=CLASSES, document=b"pdf", model="classify-latest")
    assert isinstance(result, V2ClassifyResponse)
    # The reserved `class` wire key deserializes onto the aliased `class_` field.
    assert result.classification[0].class_ == "invoice"
    assert result.classification[0].page == 0
    assert result.classification[1].suggested_class == "receipt"
    assert result.metadata.page_count == 2 and result.metadata.openapi_spec
    # `classes` rides as a JSON-encoded string form field on the multipart body.
    sent = route.calls.last.request.content
    assert b'"class": "invoice"' in sent
    assert route.calls.last.request.headers["content-type"].startswith("multipart/form-data")


@respx.mock
def test_classify_sync_omits_unset_fields_from_multipart_body() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/classify").mock(
        return_value=httpx.Response(200, json=CLASSIFY_BODY)
    )
    client.v2.classify(classes=CLASSES, document=b"pdf", document_url=None, model=None)
    sent = route.calls.last.request.content
    assert b"Omit object" not in sent
    assert b"NOT_GIVEN" not in sent
    assert b"document_url" not in sent
    assert b"model" not in sent


@respx.mock
def test_classify_sync_504_raises_v2_sync_timeout() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://api.ade.landing.ai/v2/classify").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.classify(classes=CLASSES, document=b"pdf")


def test_classify_job_create_sends_service_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, files: Any = None, options: Any = None) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "c1", "status": "pending"}

    monkeypatch.setattr(client.v2.classify_jobs, "_post", fake_post)
    client.v2.classify_jobs.create(classes=CLASSES, document=b"x", service_tier="priority")
    assert captured["body"]["service_tier"] == "priority"
    # `classes` is JSON-encoded, not a raw list, on the wire.
    assert isinstance(captured["body"]["classes"], str)
    assert json.loads(captured["body"]["classes"])[0]["class"] == "invoice"


@respx.mock
def test_classify_job_create_normalizes_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/classify/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "c1", "status": "pending"})
    )
    job = client.v2.classify_jobs.create(classes=CLASSES, document=b"pdf")
    assert isinstance(job, Job)
    assert job.job_id == "c1" and job.status is JobStatus.PENDING


@respx.mock
def test_classify_job_get_completed_has_typed_result() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/classify/jobs/c1").mock(
        return_value=httpx.Response(
            200,
            json={"job_id": "c1", "status": "completed", "created_at": 1700000000, "result": CLASSIFY_BODY},
        )
    )
    job = client.v2.classify_jobs.get("c1")
    assert job.status is JobStatus.COMPLETED
    assert isinstance(job.result, V2ClassifyResponse)
    assert job.result.classification[0].class_ == "invoice"


@respx.mock
def test_classify_job_get_failed_has_error() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/classify/jobs/c2").mock(
        return_value=httpx.Response(
            200,
            json={"job_id": "c2", "status": "failed", "error": {"code": "bad_doc", "message": "boom"}},
        )
    )
    failed = client.v2.classify_jobs.get("c2")
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None and failed.error.code == "bad_doc"
    assert failed.result is None


@respx.mock
def test_classify_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/classify/jobs").mock(
        return_value=httpx.Response(
            200,
            json={"jobs": [{"job_id": "c1", "status": "completed"}], "page": 0, "page_size": 10, "has_more": False},
        )
    )
    jobs = client.v2.classify_jobs.list()
    assert len(jobs) == 1 and jobs[0].job_id == "c1"
    assert jobs.page == 0 and jobs.page_size == 10 and jobs.has_more is False


@respx.mock
def test_classify_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.classify_jobs.get("")


@respx.mock
def test_classify_job_wait_polls_until_completed() -> None:
    client = LandingAIADE(apikey=APIKEY)
    responses = [
        httpx.Response(200, json={"job_id": "c1", "status": "processing", "progress": 0.5}),
        httpx.Response(200, json={"job_id": "c1", "status": "completed", "result": CLASSIFY_BODY}),
    ]
    respx.get("https://api.ade.landing.ai/v2/classify/jobs/c1").mock(side_effect=responses)
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    job = client.v2.classify_jobs.wait("c1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks))
    assert job.status is JobStatus.COMPLETED


@respx.mock
@pytest.mark.asyncio
async def test_async_classify_sync_ok() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/classify").mock(return_value=httpx.Response(200, json=CLASSIFY_BODY))
    result = await client.v2.classify(classes=CLASSES, document=b"pdf")
    assert isinstance(result, V2ClassifyResponse)
    assert result.classification[0].class_ == "invoice"
