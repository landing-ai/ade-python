# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from ade import Ade, AsyncAde
from ade.types import AdeParseResponse, AdeExtractResponse
from tests.utils import assert_matches_type

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestAde:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_extract(self, client: Ade) -> None:
        ade = client.ade.extract(
            schema="schema",
        )
        assert_matches_type(AdeExtractResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_extract_with_all_params(self, client: Ade) -> None:
        ade = client.ade.extract(
            schema="schema",
            markdown=b"raw file contents",
            markdown_url="markdown_url",
        )
        assert_matches_type(AdeExtractResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_extract(self, client: Ade) -> None:
        response = client.ade.with_raw_response.extract(
            schema="schema",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        ade = response.parse()
        assert_matches_type(AdeExtractResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_extract(self, client: Ade) -> None:
        with client.ade.with_streaming_response.extract(
            schema="schema",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            ade = response.parse()
            assert_matches_type(AdeExtractResponse, ade, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_parse(self, client: Ade) -> None:
        ade = client.ade.parse()
        assert_matches_type(AdeParseResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_parse_with_all_params(self, client: Ade) -> None:
        ade = client.ade.parse(
            document=b"raw file contents",
            document_url="document_url",
            split="page",
        )
        assert_matches_type(AdeParseResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_parse(self, client: Ade) -> None:
        response = client.ade.with_raw_response.parse()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        ade = response.parse()
        assert_matches_type(AdeParseResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_parse(self, client: Ade) -> None:
        with client.ade.with_streaming_response.parse() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            ade = response.parse()
            assert_matches_type(AdeParseResponse, ade, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncAde:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_extract(self, async_client: AsyncAde) -> None:
        ade = await async_client.ade.extract(
            schema="schema",
        )
        assert_matches_type(AdeExtractResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_extract_with_all_params(self, async_client: AsyncAde) -> None:
        ade = await async_client.ade.extract(
            schema="schema",
            markdown=b"raw file contents",
            markdown_url="markdown_url",
        )
        assert_matches_type(AdeExtractResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_extract(self, async_client: AsyncAde) -> None:
        response = await async_client.ade.with_raw_response.extract(
            schema="schema",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        ade = await response.parse()
        assert_matches_type(AdeExtractResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_extract(self, async_client: AsyncAde) -> None:
        async with async_client.ade.with_streaming_response.extract(
            schema="schema",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            ade = await response.parse()
            assert_matches_type(AdeExtractResponse, ade, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_parse(self, async_client: AsyncAde) -> None:
        ade = await async_client.ade.parse()
        assert_matches_type(AdeParseResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_parse_with_all_params(self, async_client: AsyncAde) -> None:
        ade = await async_client.ade.parse(
            document=b"raw file contents",
            document_url="document_url",
            split="page",
        )
        assert_matches_type(AdeParseResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_parse(self, async_client: AsyncAde) -> None:
        response = await async_client.ade.with_raw_response.parse()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        ade = await response.parse()
        assert_matches_type(AdeParseResponse, ade, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_parse(self, async_client: AsyncAde) -> None:
        async with async_client.ade.with_streaming_response.parse() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            ade = await response.parse()
            assert_matches_type(AdeParseResponse, ade, path=["response"])

        assert cast(Any, response.is_closed) is True
