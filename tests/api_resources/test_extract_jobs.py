from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tests.utils import assert_matches_type
from landingai_ade import LandingAIADE, AsyncLandingAIADE
from landingai_ade.types import (
    ExtractJobGetResponse,
    ExtractJobListResponse,
    ExtractJobCreateResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestExtractJobs:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: LandingAIADE) -> None:
        extract_job = client.extract_jobs.create(
            schema="schema",
        )
        assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: LandingAIADE) -> None:
        extract_job = client.extract_jobs.create(
            schema="schema",
            markdown=b"raw file contents",
            markdown_url="markdown_url",
            model="model",
            output_save_url="output_save_url",
            strict=True,
        )
        assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: LandingAIADE) -> None:
        response = client.extract_jobs.with_raw_response.create(
            schema="schema",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        extract_job = response.parse()
        assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: LandingAIADE) -> None:
        with client.extract_jobs.with_streaming_response.create(
            schema="schema",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            extract_job = response.parse()
            assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: LandingAIADE) -> None:
        extract_job = client.extract_jobs.list()
        assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: LandingAIADE) -> None:
        extract_job = client.extract_jobs.list(
            page=0,
            page_size=1,
            status="cancelled",
        )
        assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: LandingAIADE) -> None:
        response = client.extract_jobs.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        extract_job = response.parse()
        assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: LandingAIADE) -> None:
        with client.extract_jobs.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            extract_job = response.parse()
            assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get(self, client: LandingAIADE) -> None:
        extract_job = client.extract_jobs.get(
            "job_id",
        )
        assert_matches_type(ExtractJobGetResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_get(self, client: LandingAIADE) -> None:
        response = client.extract_jobs.with_raw_response.get(
            "job_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        extract_job = response.parse()
        assert_matches_type(ExtractJobGetResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_get(self, client: LandingAIADE) -> None:
        with client.extract_jobs.with_streaming_response.get(
            "job_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            extract_job = response.parse()
            assert_matches_type(ExtractJobGetResponse, extract_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_get(self, client: LandingAIADE) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `job_id` but received ''"):
            client.extract_jobs.with_raw_response.get(
                "",
            )


class TestAsyncExtractJobs:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncLandingAIADE) -> None:
        extract_job = await async_client.extract_jobs.create(
            schema="schema",
        )
        assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        extract_job = await async_client.extract_jobs.create(
            schema="schema",
            markdown=b"raw file contents",
            markdown_url="markdown_url",
            model="model",
            output_save_url="output_save_url",
            strict=True,
        )
        assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.extract_jobs.with_raw_response.create(
            schema="schema",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        extract_job = await response.parse()
        assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.extract_jobs.with_streaming_response.create(
            schema="schema",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            extract_job = await response.parse()
            assert_matches_type(ExtractJobCreateResponse, extract_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncLandingAIADE) -> None:
        extract_job = await async_client.extract_jobs.list()
        assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncLandingAIADE) -> None:
        extract_job = await async_client.extract_jobs.list(
            page=0,
            page_size=1,
            status="cancelled",
        )
        assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.extract_jobs.with_raw_response.list()

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        extract_job = await response.parse()
        assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.extract_jobs.with_streaming_response.list() as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            extract_job = await response.parse()
            assert_matches_type(ExtractJobListResponse, extract_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get(self, async_client: AsyncLandingAIADE) -> None:
        extract_job = await async_client.extract_jobs.get(
            "job_id",
        )
        assert_matches_type(ExtractJobGetResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_get(self, async_client: AsyncLandingAIADE) -> None:
        response = await async_client.extract_jobs.with_raw_response.get(
            "job_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        extract_job = await response.parse()
        assert_matches_type(ExtractJobGetResponse, extract_job, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_get(self, async_client: AsyncLandingAIADE) -> None:
        async with async_client.extract_jobs.with_streaming_response.get(
            "job_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            extract_job = await response.parse()
            assert_matches_type(ExtractJobGetResponse, extract_job, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_get(self, async_client: AsyncLandingAIADE) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `job_id` but received ''"):
            await async_client.extract_jobs.with_raw_response.get(
                "",
            )
