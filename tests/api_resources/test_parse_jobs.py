# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tests.utils import assert_matches_type
from landingai_ade import LandingAIADE, AsyncLandingAIADE
from landingai_ade.types import (
    ParseJobGetResponse,
    ParseJobListResponse,
    ParseJobCreateResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestParseJobs:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_create(self, client: LandingAIADE) -> None:
        parse_job = client.parse_jobs.create()
        assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: LandingAIADE) -> None:
        parse_job = client.parse_jobs.create(
            document=b"raw file contents",
            document_url="document_url",
            model="model",
            output_save_url="output_save_url",
            split="page",
        )
        assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: LandingAIADE) -> None:
        response = client.parse_jobs.with_raw_response.create()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        parse_job = response.parse()
        assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: LandingAIADE) -> None:
        with client.parse_jobs.with_streaming_response.create() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            parse_job = response.parse()
            assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_list(self, client: LandingAIADE) -> None:
        parse_job = client.parse_jobs.list()
        assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: LandingAIADE) -> None:
        parse_job = client.parse_jobs.list(
            page=0,
            page_size=1,
            status="cancelled",
        )
        assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: LandingAIADE) -> None:
        response = client.parse_jobs.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        parse_job = response.parse()
        assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: LandingAIADE) -> None:
        with client.parse_jobs.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            parse_job = response.parse()
            assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_get(self, client: LandingAIADE) -> None:
        parse_job = client.parse_jobs.get(
            "job_id",
        )
        assert_matches_type(ParseJobGetResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_get(self, client: LandingAIADE) -> None:
        response = client.parse_jobs.with_raw_response.get(
            "job_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        parse_job = response.parse()
        assert_matches_type(ParseJobGetResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_get(self, client: LandingAIADE) -> None:
        with client.parse_jobs.with_streaming_response.get(
            "job_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            parse_job = response.parse()
            assert_matches_type(ParseJobGetResponse, parse_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_path_params_get(self, client: LandingAIADE) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `job_id` but received ''"):
            client.parse_jobs.with_raw_response.get(
                "",
            )


class TestAsyncParseJobs:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncLandingAIADE) -> None:
        parse_job = await async_client.parse_jobs.create()
        assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        parse_job = await async_client.parse_jobs.create(
            document=b"raw file contents",
            document_url="document_url",
            model="model",
            output_save_url="output_save_url",
            split="page",
        )
        assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.parse_jobs.with_raw_response.create()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        parse_job = await response.parse()
        assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.parse_jobs.with_streaming_response.create() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            parse_job = await response.parse()
            assert_matches_type(ParseJobCreateResponse, parse_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncLandingAIADE) -> None:
        parse_job = await async_client.parse_jobs.list()
        assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        parse_job = await async_client.parse_jobs.list(
            page=0,
            page_size=1,
            status="cancelled",
        )
        assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.parse_jobs.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        parse_job = await response.parse()
        assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.parse_jobs.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            parse_job = await response.parse()
            assert_matches_type(ParseJobListResponse, parse_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_get(self, async_client: AsyncLandingAIADE) -> None:
        parse_job = await async_client.parse_jobs.get(
            "job_id",
        )
        assert_matches_type(ParseJobGetResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_get(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.parse_jobs.with_raw_response.get(
            "job_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        parse_job = await response.parse()
        assert_matches_type(ParseJobGetResponse, parse_job, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_get(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.parse_jobs.with_streaming_response.get(
            "job_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            parse_job = await response.parse()
            assert_matches_type(ParseJobGetResponse, parse_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_path_params_get(self, async_client: AsyncLandingAIADE) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `job_id` but received ''"):
            await async_client.parse_jobs.with_raw_response.get(
                "",
            )
