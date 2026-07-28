from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2GroundResult

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
def test_ground_accepts_pydantic_structure() -> None:
    # `structure` may be passed as a pydantic model (e.g. a parse response's
    # `.structure`); it is coerced to a dict on the wire.
    from landingai_ade.types.v2 import V2ParseStructure

    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/ground").mock(return_value=httpx.Response(200, json=GROUND_BODY))
    client.v2.ground(extraction_metadata=EXTRACTION_METADATA, structure=V2ParseStructure())
    req = json.loads(route.calls.last.request.content)
    assert req["structure"]["type"] == "document"
