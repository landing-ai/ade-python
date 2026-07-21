from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import JobStatus, V2GroundResult
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"

EXTRACTION_METADATA: Dict[str, Any] = {
    "invoice_number": {"value": "INV-042", "ranges": [{"start": 13, "end": 31}]},
}
STRUCTURE: Dict[str, Any] = {"type": "document", "children": []}
GROUND_BODY: Dict[str, Any] = {
    "grounding": {
        "invoice_number": [
            {
                "block_id": "text-1",
                "type": "text",
                "grounding": {"page": 1, "range": {"start": 13, "end": 31}},
            }
        ]
    },
    "metadata": {"job_id": "ground-1", "duration_ms": 5},
}


@respx.mock
def test_ground_sync_json_body() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/ground").mock(return_value=httpx.Response(200, json=GROUND_BODY))
    result = client.v2.ground(extraction_metadata=EXTRACTION_METADATA, structure=STRUCTURE)
    assert isinstance(result, V2GroundResult)
    assert result.metadata.job_id == "ground-1"
    assert "invoice_number" in result.grounding
    req = json.loads(route.calls.last.request.content)
    assert req["extraction_metadata"]["invoice_number"]["value"] == "INV-042"
    assert req["structure"]["type"] == "document"
    assert route.calls.last.request.headers["content-type"].startswith("application/json")


@respx.mock
def test_ground_sync_parses_billing_metadata() -> None:
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(GROUND_BODY)
    metadata: Dict[str, Any] = dict(GROUND_BODY["metadata"])
    metadata["billing"] = {"service_tier": "priority", "total_credits": 2.5}
    body["metadata"] = metadata
    respx.post("https://api.ade.landing.ai/v2/ground").mock(return_value=httpx.Response(200, json=body))
    result = client.v2.ground(extraction_metadata=EXTRACTION_METADATA, structure=STRUCTURE)
    assert result.metadata.billing is not None
    assert result.metadata.billing.service_tier == "priority"
    assert result.metadata.billing.total_credits == 2.5


@respx.mock
def test_ground_sync_504() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://api.ade.landing.ai/v2/ground").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.ground(extraction_metadata=EXTRACTION_METADATA, structure=STRUCTURE)


@respx.mock
def test_ground_job_create_and_get() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/ground/jobs").mock(
        return_value=httpx.Response(
            202, json={"job_id": "ground-1", "status": "pending", "created_at": "2026-01-01T00:00:00Z"}
        )
    )
    job = client.v2.ground_jobs.create(extraction_metadata=EXTRACTION_METADATA, structure=STRUCTURE)
    assert job.job_id == "ground-1" and job.status is JobStatus.PENDING

    respx.get("https://api.ade.landing.ai/v2/ground/jobs/ground-1").mock(
        return_value=httpx.Response(
            200,
            json={
                "job_id": "ground-1",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:09Z",
                "result": GROUND_BODY,
            },
        )
    )
    done = client.v2.ground_jobs.get("ground-1")
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2GroundResult)
    assert done.result.metadata.job_id == "ground-1"


@respx.mock
def test_ground_job_create_sends_output_save_url() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/ground/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "ground-2", "status": "pending"})
    )
    client.v2.ground_jobs.create(
        extraction_metadata=EXTRACTION_METADATA,
        structure=STRUCTURE,
        output_save_url="https://example.com/put",
    )
    req = json.loads(route.calls.last.request.content)
    assert req["output_save_url"] == "https://example.com/put"


@respx.mock
def test_ground_job_get_failed_maps_error_object() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/ground/jobs/g2").mock(
        return_value=httpx.Response(
            200, json={"job_id": "g2", "status": "failed", "error": {"code": "internal_error", "message": "boom"}}
        )
    )
    job = client.v2.ground_jobs.get("g2")
    assert job.status is JobStatus.FAILED and job.error is not None and job.error.code == "internal_error"


@respx.mock
def test_ground_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.ground_jobs.get("")


@respx.mock
def test_ground_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/ground/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [{"job_id": "ground-1", "status": "completed"}],
                "page": 0,
                "page_size": 10,
                "has_more": True,
            },
        )
    )
    jobs = client.v2.ground_jobs.list()
    assert len(jobs) == 1 and jobs[0].job_id == "ground-1" and jobs[0].status is JobStatus.COMPLETED
    assert jobs.has_more is True
    assert jobs.page == 0
    assert jobs.page_size == 10


@respx.mock
def test_ground_accepts_pydantic_structure() -> None:
    # `structure` may be passed as a pydantic model (e.g. a parse response's
    # `.structure`); it is coerced to a dict on the wire.
    from landingai_ade.types.v2 import V2ParseStructure

    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/ground").mock(return_value=httpx.Response(200, json=GROUND_BODY))
    client.v2.ground(extraction_metadata=EXTRACTION_METADATA, structure=V2ParseStructure())
    req = json.loads(route.calls.last.request.content)
    assert req["structure"]["type"] == "document"
