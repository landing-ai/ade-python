from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2ParseResponse
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
