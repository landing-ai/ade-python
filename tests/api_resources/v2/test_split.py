from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import V2SplitResponse

APIKEY = "My Apikey"

SPLIT_CLASS = [
    {"name": "invoice", "description": "A billing document", "identifier": "invoice_number"},
    {"name": "receipt"},
]

SPLIT_BODY: Dict[str, Any] = {
    "splits": [
        {
            "classification": "invoice",
            "identifier": "INV-042",
            "markdowns": ["# Invoice", "## Line items"],
            "pages": [0, 1],
        },
        {
            "classification": "receipt",
            "identifier": None,
            "markdowns": ["# Receipt"],
            "pages": [2],
        },
    ],
    "metadata": {
        "filename": "doc.md",
        "org_id": None,
        "page_count": 3,
        "duration_ms": 11,
        "credit_usage": 0.1,
        "job_id": "split-1",
        "version": "split-20251105",
    },
}


@respx.mock
def test_split_sync_routes_to_v2_and_encodes_split_class() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v2/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    result = client.v2.split(split_class=SPLIT_CLASS, markdown="# Hello", model="split-latest")
    assert isinstance(result, V2SplitResponse)
    assert result.splits[0].classification == "invoice"
    assert result.splits[0].identifier == "INV-042"
    assert result.splits[0].pages == [0, 1]
    # `identifier`/`org_id` are required-but-nullable; a null wire value survives.
    assert result.splits[1].identifier is None
    assert result.metadata.org_id is None
    assert result.metadata.job_id == "split-1"
    # `split_class` rides as a JSON-encoded string form field on the multipart body.
    sent = route.calls.last.request.content
    assert b'"name": "invoice"' in sent
    assert route.calls.last.request.headers["content-type"].startswith("multipart/form-data")


@respx.mock
def test_split_sync_omits_unset_fields_from_multipart_body() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    client.v2.split(split_class=SPLIT_CLASS, markdown="# Hello", markdown_url=None, model=None)
    sent = route.calls.last.request.content
    assert b"Omit object" not in sent
    assert b"NOT_GIVEN" not in sent
    assert b"markdown_url" not in sent
    assert b'name="model"' not in sent


def test_split_body_encodes_split_class_as_json() -> None:
    from landingai_ade.resources.v2.split import _build_split_body

    body = _build_split_body(SPLIT_CLASS, "# Hello", None, None)
    assert isinstance(body["split_class"], str)
    assert json.loads(body["split_class"])[0]["name"] == "invoice"
    # Explicit `None` optional fields never leak into the multipart body.
    assert "markdown_url" not in body
    assert "model" not in body


@respx.mock
def test_split_save_to_writes_file(tmp_path: Path) -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    client.v2.split(split_class=SPLIT_CLASS, markdown="# Hello", save_to=str(tmp_path))
    written = list(tmp_path.glob("*.json"))
    assert written and json.loads(written[0].read_text())["metadata"]["job_id"] == "split-1"


@respx.mock
@pytest.mark.asyncio
async def test_async_split_sync_ok() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/split").mock(return_value=httpx.Response(200, json=SPLIT_BODY))
    result = await client.v2.split(split_class=SPLIT_CLASS, markdown="# Hello")
    assert isinstance(result, V2SplitResponse)
    assert result.splits[0].classification == "invoice"
