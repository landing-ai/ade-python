# Regression tests for https://github.com/landing-ai/ade-python/issues/76:
# split_class must be sent as a single JSON string form field, not flattened
# into split_class[0][name]-style multipart fields.
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from landingai_ade import LandingAIADE, AsyncLandingAIADE

SPLIT_CLASS = [
    {"name": "Bank Statement", "description": "Bank account activity summary."},
    {"name": "Pay Stub", "description": "Employee earnings detail.", "identifier": "Pay Stub Date"},
]


def test_sync_split_class_sent_as_json_string() -> None:
    with LandingAIADE(apikey="test-key", base_url="http://localhost") as client:
        with patch.object(client, "post", return_value=MagicMock()) as post:
            client.split(split_class=SPLIT_CLASS, markdown="some markdown")

    body = post.call_args.kwargs["body"]
    assert body["split_class"] == json.dumps(SPLIT_CLASS)
    assert json.loads(body["split_class"]) == SPLIT_CLASS


def test_sync_split_class_json_string_passed_through() -> None:
    # An already JSON-encoded split_class (documented in the README) must not be re-serialized.
    encoded = json.dumps(SPLIT_CLASS)
    with LandingAIADE(apikey="test-key", base_url="http://localhost") as client:
        with patch.object(client, "post", return_value=MagicMock()) as post:
            client.split(split_class=encoded, markdown="some markdown")

    assert post.call_args.kwargs["body"]["split_class"] == encoded


@pytest.mark.asyncio
async def test_async_split_class_json_string_passed_through() -> None:
    encoded = json.dumps(SPLIT_CLASS)
    async with AsyncLandingAIADE(apikey="test-key", base_url="http://localhost") as client:
        with patch.object(client, "post", new_callable=AsyncMock, return_value=MagicMock()) as post:
            await client.split(split_class=encoded, markdown="some markdown")

    assert post.call_args.kwargs["body"]["split_class"] == encoded


@pytest.mark.asyncio
async def test_async_split_class_sent_as_json_string() -> None:
    async with AsyncLandingAIADE(apikey="test-key", base_url="http://localhost") as client:
        with patch.object(client, "post", new_callable=AsyncMock, return_value=MagicMock()) as post:
            await client.split(split_class=SPLIT_CLASS, markdown="some markdown")

    body = post.call_args.kwargs["body"]
    assert body["split_class"] == json.dumps(SPLIT_CLASS)
    assert json.loads(body["split_class"]) == SPLIT_CLASS
