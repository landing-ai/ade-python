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
    "metadata": {"job_id": "bs1", "duration_ms": 5, "openapi_spec": "https://api.ade.landing.ai/openapi.json"},
}


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


@respx.mock
def test_build_schema_sync_json_body_with_prompt() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    result = client.v2.build_schema(prompt="Extract the revenue figure")
    assert isinstance(result, V2BuildSchemaResponse)
    assert json.loads(result.extraction_schema)["type"] == "object"
    assert result.metadata.job_id == "bs1"
    req = json.loads(route.calls.last.request.content)
    assert req["prompt"] == "Extract the revenue figure"
    assert route.calls.last.request.headers["content-type"].startswith("application/json")


@respx.mock
def test_build_schema_sync_json_markdowns_and_urls() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    client.v2.build_schema(markdowns=["# doc one", "# doc two"], markdown_urls=["https://x/y.md"])
    req = json.loads(route.calls.last.request.content)
    # Inline-string markdowns stay a JSON array (no file → JSON, not multipart).
    assert req["markdowns"] == ["# doc one", "# doc two"]
    assert req["markdown_urls"] == ["https://x/y.md"]
    assert route.calls.last.request.headers["content-type"].startswith("application/json")


@respx.mock
def test_build_schema_sync_coerces_schema_to_json_string() -> None:
    # `schema` (a pydantic model / dict / JSON string) is coerced to a JSON Schema
    # and sent as a JSON-encoded *string* per the contract's string-typed field.
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    client.v2.build_schema(schema=Invoice)
    req = json.loads(route.calls.last.request.content)
    assert isinstance(req["schema"], str)
    parsed = json.loads(req["schema"])
    assert parsed["type"] == "object" and "revenue" in parsed["properties"]


@respx.mock
def test_build_schema_sync_multipart_when_markdown_is_a_file() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    client.v2.build_schema(markdowns=[b"# uploaded doc"], prompt="go")
    assert route.calls.last.request.headers["content-type"].startswith("multipart/form-data")
    sent = route.calls.last.request.content
    assert b'name="markdowns"' in sent
    assert b"# uploaded doc" in sent
    assert b'name="prompt"' in sent


@respx.mock
def test_build_schema_sync_multipart_serializes_urls_as_json_string() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    client.v2.build_schema(markdowns=[b"# doc"], markdown_urls=["https://x/y.md"])
    sent = route.calls.last.request.content
    # markdown_urls is a JSON-serialized string form field in multipart.
    assert b'["https://x/y.md"]' in sent


@respx.mock
def test_build_schema_requires_at_least_one_input() -> None:
    # No route registered: @respx.mock keeps this hermetic so a regression fails
    # loudly on an unmocked request instead of hitting the network.
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError, match="at least one"):
        client.v2.build_schema()


@respx.mock
def test_build_schema_job_create_requires_at_least_one_input() -> None:
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
    assert json.loads(done.result.extraction_schema)["type"] == "object"


def test_build_schema_job_create_sends_service_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, files: Any = None, options: Any = None) -> Any:  # noqa: ARG001
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
                "jobs": [{"job_id": "bs1", "status": "completed"}],
                "page": 0,
                "page_size": 10,
                "has_more": True,
            },
        )
    )
    jobs = client.v2.build_schema_jobs.list()
    assert len(jobs) == 1 and jobs[0].job_id == "bs1" and jobs[0].status is JobStatus.COMPLETED
    assert jobs.has_more is True and jobs.page == 0 and jobs.page_size == 10


@respx.mock
def test_build_schema_job_list_status_none_omits_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/extract/build-schema/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.build_schema_jobs.list(status=None)
    assert "status" not in route.calls.last.request.url.params


@respx.mock
@pytest.mark.asyncio
async def test_async_build_schema_sync_ok() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/extract/build-schema").mock(
        return_value=httpx.Response(200, json=BUILD_SCHEMA_BODY)
    )
    result = await client.v2.build_schema(prompt="x")
    assert isinstance(result, V2BuildSchemaResponse)
    assert result.metadata.job_id == "bs1"


@respx.mock
@pytest.mark.asyncio
async def test_async_build_schema_job_create_and_get() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/extract/build-schema/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "bs1", "status": "pending"})
    )
    respx.get("https://api.ade.landing.ai/v2/extract/build-schema/jobs/bs1").mock(
        return_value=httpx.Response(200, json={"job_id": "bs1", "status": "completed", "result": BUILD_SCHEMA_BODY})
    )
    created = await client.v2.build_schema_jobs.create(prompt="x")
    assert created.status is JobStatus.PENDING
    fetched = await client.v2.build_schema_jobs.get("bs1")
    assert fetched.status is JobStatus.COMPLETED
    assert isinstance(fetched.result, V2BuildSchemaResponse)
