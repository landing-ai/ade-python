from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx
import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2ExtractResult
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
