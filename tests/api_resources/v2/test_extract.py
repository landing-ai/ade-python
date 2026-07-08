from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx
import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import JobStatus, V2ExtractResult
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
EXTRACT_BODY: Dict[str, Any] = {
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
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
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
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(schema={"type": "object", "properties": {}}, markdown_url="https://x/y.md", strict=True)
    req = json.loads(route.calls.last.request.content)
    assert req["options"]["strict"] is True
    assert req["markdown_url"] == "https://x/y.md"


@respx.mock
def test_extract_sync_504() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://api.ade.landing.ai/v2/extract").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.extract(schema={"type": "object"}, markdown="x")


@respx.mock
def test_extract_job_create_and_get() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "e1", "status": "pending", "created_at": "2026-01-01T00:00:00Z"})
    )
    job = client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x", priority="priority")
    assert job.job_id == "e1" and job.status is JobStatus.PENDING

    respx.get("https://api.ade.landing.ai/v2/extract/jobs/e1").mock(
        return_value=httpx.Response(
            200,
            json={"job_id": "e1", "status": "completed", "created_at": "2026-01-01T00:00:00Z",
                  "completed_at": "2026-01-01T00:00:09Z", "result": EXTRACT_BODY},
        )
    )
    done = client.v2.extract_jobs.get("e1")
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2ExtractResult)
    assert done.result.metadata.version == "extract-1"


@respx.mock
def test_extract_job_get_failed_maps_error_object() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/jobs/e2").mock(
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
    respx.get("https://api.ade.landing.ai/v2/extract/jobs/e3").mock(
        return_value=httpx.Response(200, json={"job_id": "e3", "status": "failed", "error": {"code": "x", "message": "no"}})
    )
    with pytest.raises(JobFailedError):
        client.v2.extract_jobs.wait("e3", timeout=5, poll_interval=0.01, raise_on_failure=True,
                                    _monotonic=lambda: 0.0)


@respx.mock
def test_extract_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.extract_jobs.get("")


@respx.mock
def test_extract_job_list_status_none_omits_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.extract_jobs.list(status=None)
    assert "status" not in route.calls.last.request.url.params


def test_extract_job_list_status_none_excluded_from_query_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test at the query-dict level: the underlying querystring
    # encoder happens to drop `None`-valued params when serializing to a URL,
    # which would mask this bug in an end-to-end/respx assertion. Capture the
    # dict handed to `options["params"]` directly so a regression is caught
    # even before it reaches that encoder.
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_get(path: str, *, cast_to: Any, options: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["params"] = dict(options or {}).get("params", {})
        return {"jobs": [], "has_more": False}

    monkeypatch.setattr(client.v2.extract_jobs, "_get", fake_get)

    client.v2.extract_jobs.list(status=None)
    assert "status" not in captured["params"]

    client.v2.extract_jobs.list(status="completed")
    assert captured["params"].get("status") == "completed"


@respx.mock
def test_extract_job_list_status_given_includes_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.extract_jobs.list(status="completed")
    assert route.calls.last.request.url.params["status"] == "completed"


@respx.mock
def test_extract_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [{"job_id": "e1", "status": "completed"}],
                "page": 0,
                "page_size": 10,
                "has_more": True,
            },
        )
    )
    jobs = client.v2.extract_jobs.list()
    assert len(jobs) == 1 and jobs[0].job_id == "e1" and jobs[0].status is JobStatus.COMPLETED
    assert jobs.has_more is True
    assert jobs.page == 0
    assert jobs.page_size == 10
