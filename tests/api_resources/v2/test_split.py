from __future__ import annotations

from typing import Any, Dict

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2SplitResponse

APIKEY = "My Apikey"

SPLIT_CLASS = [
    {"name": "invoice", "description": "A billing invoice", "identifier": "invoice_number"},
    {"name": "receipt"},
]

SPLIT_BODY: Dict[str, Any] = {
    "splits": [
        {
            "classification": "invoice",
            "identifier": "INV-042",
            "markdowns": ["# Invoice\n", "line items\n"],
            "pages": [0, 1],
        },
        {
            "classification": "receipt",
            "identifier": None,
            "markdowns": ["# Receipt\n"],
            "pages": [2],
        },
    ],
    "metadata": {
        "credit_usage": 0.1,
        "duration_ms": 12,
        "filename": "doc.md",
        "job_id": "split-1",
        "org_id": None,
        "page_count": 3,
        "version": "split-20251105",
    },
}


@respx.mock
def test_split_sync_routes_to_v1_and_sends_split_class_json() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v1/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    result = client.v2.split(split_class=SPLIT_CLASS, markdown="# Invoice")
    assert isinstance(result, V2SplitResponse)
    assert len(result.splits) == 2
    assert result.splits[0].classification == "invoice"
    assert result.splits[0].identifier == "INV-042"
    assert result.splits[0].pages == [0, 1]
    # An identifier that is null on the wire deserializes to None.
    assert result.splits[1].identifier is None
    assert result.metadata.version == "split-20251105"
    assert result.metadata.org_id is None
    # `split_class` must be sent as a JSON-encoded string form field.
    sent = route.calls.last.request.content
    assert b'name="split_class"' in sent
    assert b'"name": "invoice"' in sent


@respx.mock
def test_split_sync_omits_unset_and_none_fields() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v1/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    client.v2.split(split_class=SPLIT_CLASS, markdown_url="https://example.com/doc.md", model=None)
    sent = route.calls.last.request.content
    assert b"Omit object" not in sent
    assert b"NOT_GIVEN" not in sent
    assert b'name="markdown"' not in sent
    assert b'name="model"' not in sent
    assert b'name="markdown_url"' in sent


@respx.mock
@pytest.mark.asyncio
async def test_async_split_sync_ok() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY, environment="production")
    respx.post("https://api.ade.landing.ai/v1/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    result = await client.v2.split(split_class=SPLIT_CLASS, markdown="# Invoice")
    assert isinstance(result, V2SplitResponse)
    assert result.splits[0].markdowns == ["# Invoice\n", "line items\n"]
