from __future__ import annotations

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE, AsyncLandingAIADE

APIKEY = "My Apikey"


@respx.mock
def test_files_upload_returns_ref_and_hits_v2_host() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://aide.landing.ai/v1/files").mock(
        return_value=httpx.Response(200, json={"file_ref": "fr_1"})
    )
    ref = client.v2.files.upload(file=b"markdown bytes")
    assert route.called
    assert route.calls.last.request.headers["authorization"] == "Bearer My Apikey"
    assert ref == "fr_1"


@respx.mock
@pytest.mark.asyncio
async def test_async_files_upload() -> None:
    client = AsyncLandingAIADE(apikey=APIKEY, environment="staging")
    respx.post("https://aide.staging.landing.ai/v1/files").mock(
        return_value=httpx.Response(200, json={"file_ref": "fr_2"})
    )
    ref = await client.v2.files.upload(file=b"bytes")
    assert ref == "fr_2"
