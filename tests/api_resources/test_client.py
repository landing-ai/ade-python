# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tests.utils import assert_matches_type
from landingai_ade import LandingAIADE, AsyncLandingAIADE
from landingai_ade.types import (
    ParseResponse,
    SplitResponse,
    ExtractResponse,
    ExtractBuildResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestClient:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_extract(self, client: LandingAIADE) -> None:
        client_ = client.extract(
            schema="schema",
        )
        assert_matches_type(ExtractResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_extract_with_all_params(self, client: LandingAIADE) -> None:
        client_ = client.extract(
            schema="schema",
            markdown=b"Example data",
            markdown_url="markdown_url",
            model="model",
            strict=True,
        )
        assert_matches_type(ExtractResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_extract(self, client: LandingAIADE) -> None:
        response = client.with_raw_response.extract(
            schema="schema",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client_ = response.parse()
        assert_matches_type(ExtractResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_extract(self, client: LandingAIADE) -> None:
        with client.with_streaming_response.extract(
            schema="schema",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client_ = response.parse()
            assert_matches_type(ExtractResponse, client_, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_extract_build(self, client: LandingAIADE) -> None:
        client_ = client.extract_build()
        assert_matches_type(ExtractBuildResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_extract_build_with_all_params(self, client: LandingAIADE) -> None:
        client_ = client.extract_build(
            markdown_urls=["string"],
            markdowns=[b"Example data"],
            model="model",
            prompt="prompt",
            schema="schema",
        )
        assert_matches_type(ExtractBuildResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_extract_build(self, client: LandingAIADE) -> None:
        response = client.with_raw_response.extract_build()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client_ = response.parse()
        assert_matches_type(ExtractBuildResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_extract_build(self, client: LandingAIADE) -> None:
        with client.with_streaming_response.extract_build() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client_ = response.parse()
            assert_matches_type(ExtractBuildResponse, client_, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_parse(self, client: LandingAIADE) -> None:
        client_ = client.parse()
        assert_matches_type(ParseResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_parse_with_all_params(self, client: LandingAIADE) -> None:
        client_ = client.parse(
            custom_prompts={"figure": "figure"},
            document=b"Example data",
            document_url="document_url",
            model="model",
            password="password",
            split="page",
        )
        assert_matches_type(ParseResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_parse(self, client: LandingAIADE) -> None:
        response = client.with_raw_response.parse()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client_ = response.parse()
        assert_matches_type(ParseResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_parse(self, client: LandingAIADE) -> None:
        with client.with_streaming_response.parse() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client_ = response.parse()
            assert_matches_type(ParseResponse, client_, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_split(self, client: LandingAIADE) -> None:
        client_ = client.split(
            split_class=[{"name": "name"}],
        )
        assert_matches_type(SplitResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_split_with_all_params(self, client: LandingAIADE) -> None:
        client_ = client.split(
            split_class=[
                {
                    "name": "name",
                    "description": "description",
                    "identifier": "identifier",
                }
            ],
            markdown=b"Example data",
            markdown_url="markdown_url",
            model="model",
        )
        assert_matches_type(SplitResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_split(self, client: LandingAIADE) -> None:
        response = client.with_raw_response.split(
            split_class=[{"name": "name"}],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client_ = response.parse()
        assert_matches_type(SplitResponse, client_, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_split(self, client: LandingAIADE) -> None:
        with client.with_streaming_response.split(
            split_class=[{"name": "name"}],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client_ = response.parse()
            assert_matches_type(SplitResponse, client_, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncClient:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_extract(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.extract(
            schema="schema",
        )
        assert_matches_type(ExtractResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_extract_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.extract(
            schema="schema",
            markdown=b"Example data",
            markdown_url="markdown_url",
            model="model",
            strict=True,
        )
        assert_matches_type(ExtractResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_extract(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.with_raw_response.extract(
            schema="schema",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client = await response.parse()
        assert_matches_type(ExtractResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_extract(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.with_streaming_response.extract(
            schema="schema",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client = await response.parse()
            assert_matches_type(ExtractResponse, client, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_extract_build(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.extract_build()
        assert_matches_type(ExtractBuildResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_extract_build_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.extract_build(
            markdown_urls=["string"],
            markdowns=[b"Example data"],
            model="model",
            prompt="prompt",
            schema="schema",
        )
        assert_matches_type(ExtractBuildResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_extract_build(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.with_raw_response.extract_build()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client = await response.parse()
        assert_matches_type(ExtractBuildResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_extract_build(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.with_streaming_response.extract_build() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client = await response.parse()
            assert_matches_type(ExtractBuildResponse, client, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_parse(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.parse()
        assert_matches_type(ParseResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_parse_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.parse(
            custom_prompts={"figure": "figure"},
            document=b"Example data",
            document_url="document_url",
            model="model",
            password="password",
            split="page",
        )
        assert_matches_type(ParseResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_parse(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.with_raw_response.parse()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client = await response.parse()
        assert_matches_type(ParseResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_parse(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.with_streaming_response.parse() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client = await response.parse()
            assert_matches_type(ParseResponse, client, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_split(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.split(
            split_class=[{"name": "name"}],
        )
        assert_matches_type(SplitResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_split_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        client = await async_client.split(
            split_class=[
                {
                    "name": "name",
                    "description": "description",
                    "identifier": "identifier",
                }
            ],
            markdown=b"Example data",
            markdown_url="markdown_url",
            model="model",
        )
        assert_matches_type(SplitResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_split(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.with_raw_response.split(
            split_class=[{"name": "name"}],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        client = await response.parse()
        assert_matches_type(SplitResponse, client, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_split(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.with_streaming_response.split(
            split_class=[{"name": "name"}],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            client = await response.parse()
            assert_matches_type(SplitResponse, client, path=["response"])

        assert cast(Any, response.is_closed) is True
