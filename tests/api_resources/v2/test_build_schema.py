from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx
import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import JobStatus, V2BuildSchemaResponse
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
BUILD_SCHEMA_BODY: Dict[str, Any] = {
    "extraction_schema": json.dumps({"type": "object", "properties": {"revenue": {"type": "string"}}}),
    "metadata": {"job_id": "bs1", "duration_ms": 5, "openapi_spec": "https://x/openapi.json"},
}


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


@respx.mock
def test_build_schema_sync_json_body_with_pydantic_schema() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    result = client.v2.build_schema(schema=Invoice, prompt="tighten it up")
    assert isinstance(result, V2BuildSchemaResponse)
    assert isinstance(result.extraction_schema, str)
    assert result.metadata.job_id == "bs1"
    req = json.loads(route.calls.last.request.content)
    # `schema` is sent as a JSON *string* (VTRA parity), not an object.
    assert isinstance(req["schema"], str)
    assert json.loads(req["schema"])["properties"]["revenue"]["type"] == "string"
    assert req["prompt"] == "tighten it up"
    assert route.calls.last.request.headers["content-type"].startswith("application/json")


@respx.mock
def test_build_schema_sync_sends_markdowns_and_urls() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    client.v2.build_schema(markdowns=["# a", "# b"], markdown_urls=["https://x/y.md"])
    req = json.loads(route.calls.last.request.content)
    assert req["markdowns"] == ["# a", "# b"]
    assert req["markdown_urls"] == ["https://x/y.md"]


@respx.mock
def test_build_schema_requires_a_source() -> None:
    # api.md documents the contract: provide at least one of markdowns /
    # markdown_urls / prompt / schema. No route is registered, so a regression in
    # the guard would fail loudly on an unmocked request.
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError, match="at least one"):
        client.v2.build_schema()


@respx.mock
def test_build_schema_job_create_requires_a_source() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError, match="at least one"):
        client.v2.build_schema_jobs.create()


@respx.mock
def test_build_schema_sync_504() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(504, json={"detail": "x"})
    )
    with pytest.raises(V2SyncTimeoutError):
        client.v2.build_schema(prompt="x")


@respx.mock
def test_build_schema_sync_parses_billing_and_warnings() -> None:
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(BUILD_SCHEMA_BODY)
    metadata: Dict[str, Any] = dict(BUILD_SCHEMA_BODY["metadata"])
    metadata["billing"] = {"service_tier": "priority", "total_credits": 3.5}
    metadata["warnings"] = [{"code": "nonconformant_schema", "msg": "heads up"}]
    body["metadata"] = metadata
    respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(return_value=httpx.Response(200, json=body))
    result = client.v2.build_schema(prompt="x")
    assert result.metadata.billing is not None
    assert result.metadata.billing.service_tier == "priority"
    assert result.metadata.warnings is not None
    assert result.metadata.warnings[0].code == "nonconformant_schema"
    assert result.metadata.warnings[0].msg == "heads up"


@respx.mock
def test_build_schema_job_create_and_get() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/extract/build-schema/jobs").mock(
        return_value=httpx.Response(
            202, json={"job_id": "bs1", "status": "pending", "created_at": "2026-01-01T00:00:00Z"}
        )
    )
    job = client.v2.build_schema_jobs.create(prompt="x", service_tier="priority")
    assert job.job_id == "bs1" and job.status is JobStatus.PENDING

    respx.get("https://api.ade.landing.ai/v2/extract/build-schema/jobs/bs1").mock(
        return_value=httpx.Response(
            200,
            json={
                "job_id": "bs1",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:09Z",
                "result": BUILD_SCHEMA_BODY,
            },
        )
    )
    done = client.v2.build_schema_jobs.get("bs1")
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2BuildSchemaResponse)
    assert done.result.metadata.job_id == "bs1"


def test_build_schema_job_create_sends_service_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, options: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "bs1", "status": "pending"}

    monkeypatch.setattr(client.v2.build_schema_jobs, "_post", fake_post)
    client.v2.build_schema_jobs.create(prompt="x", service_tier="priority")
    assert captured["body"]["service_tier"] == "priority"


@respx.mock
def test_build_schema_job_get_failed_maps_error_object() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/build-schema/jobs/bs2").mock(
        return_value=httpx.Response(
            200, json={"job_id": "bs2", "status": "failed", "error": {"code": "internal_error", "message": "boom"}}
        )
    )
    job = client.v2.build_schema_jobs.get("bs2")
    assert job.status is JobStatus.FAILED and job.error is not None and job.error.code == "internal_error"


@respx.mock
def test_build_schema_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.build_schema_jobs.get("")


@respx.mock
def test_build_schema_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/build-schema/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [{"job_id": "bs1", "status": "completed", "failure_reason": None}],
                "page": 0,
                "page_size": 10,
                "has_more": True,
            },
        )
    )
    jobs = client.v2.build_schema_jobs.list()
    assert len(jobs) == 1 and jobs[0].job_id == "bs1" and jobs[0].status is JobStatus.COMPLETED
    assert jobs.has_more is True
    assert jobs.page == 0
    assert jobs.page_size == 10
