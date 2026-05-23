# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Mapping, Optional, cast
from typing_extensions import Literal

import httpx

from ..types import extract_job_list_params, extract_job_create_params
from .._files import deepcopy_with_paths
from .._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from .._utils import extract_files, path_template, maybe_transform, async_maybe_transform
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.extract_job_get_response import ExtractJobGetResponse
from ..types.extract_job_list_response import ExtractJobListResponse
from ..types.extract_job_create_response import ExtractJobCreateResponse

__all__ = ["ExtractJobsResource", "AsyncExtractJobsResource"]


class ExtractJobsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ExtractJobsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/landing-ai/ade-python#accessing-raw-response-data-eg-headers
        """
        return ExtractJobsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ExtractJobsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/landing-ai/ade-python#with_streaming_response
        """
        return ExtractJobsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        schema: str,
        markdown: Optional[FileTypes | str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        strict: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractJobCreateResponse:
        """
        Extract structured data from Markdown asynchronously.

        This endpoint creates a job that handles the extraction for both large documents
        and complex schemas.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/extract/jobs`.

        Args:
          schema: JSON schema for field extraction. This schema determines what key-values
              pairs are extracted from the Markdown. The schema must be a valid JSON
              object and will be validated before processing the document.

          markdown: The Markdown file or Markdown content to extract data from.

          markdown_url: The URL to the Markdown file to extract data from.

          model: The version of the model to use for extraction. Use `extract-latest` to
              use the latest version.

          output_save_url: If zero data retention (ZDR) is enabled, you must enter a URL for the
              extracted output to be saved to. When ZDR is enabled, the extracted content
              will not be in the API response.

          strict: If True, reject schemas with unsupported fields (HTTP 422). If False, prune
              unsupported fields and continue. Only applies to extract versions that
              support schema validation.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_with_paths(
            {
                "schema": schema,
                "markdown": markdown,
                "markdown_url": markdown_url,
                "model": model,
                "output_save_url": output_save_url,
                "strict": strict,
            },
            [["markdown"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            "/v1/ade/extract/jobs",
            body=maybe_transform(body, extract_job_create_params.ExtractJobCreateParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExtractJobCreateResponse,
        )

    def get(
        self,
        job_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractJobGetResponse:
        """
        Retrieve the status and results of an async extract job.

        Args:
          job_id: The unique identifier of the extract job.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get(
            path_template("/v1/ade/extract/jobs/{job_id}", job_id=job_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExtractJobGetResponse,
        )

    def list(
        self,
        *,
        ending_before: Optional[str] | NotGiven = not_given,
        limit: int | NotGiven = not_given,
        starting_after: Optional[str] | NotGiven = not_given,
        status: Optional[Literal["pending", "processing", "completed", "failed", "cancelled"]] | NotGiven = not_given,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractJobListResponse:
        """
        List async extract jobs with optional filters.

        Args:
          ending_before: A cursor to retrieve results before a specific job ID.

          limit: Number of results to return (default 10, max 100).

          starting_after: A cursor to retrieve results after a specific job ID.

          status: Filter by job status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get(
            "/v1/ade/extract/jobs",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "ending_before": ending_before,
                        "limit": limit,
                        "starting_after": starting_after,
                        "status": status,
                    },
                    extract_job_list_params.ExtractJobListParams,
                ),
            ),
            cast_to=ExtractJobListResponse,
        )


class AsyncExtractJobsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncExtractJobsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/landing-ai/ade-python#accessing-raw-response-data-eg-headers
        """
        return AsyncExtractJobsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncExtractJobsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/landing-ai/ade-python#with_streaming_response
        """
        return AsyncExtractJobsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        schema: str,
        markdown: Optional[FileTypes | str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        strict: bool | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractJobCreateResponse:
        """
        Extract structured data from Markdown asynchronously.

        This endpoint creates a job that handles the extraction for both large documents
        and complex schemas.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/extract/jobs`.

        Args:
          schema: JSON schema for field extraction. This schema determines what key-values
              pairs are extracted from the Markdown. The schema must be a valid JSON
              object and will be validated before processing the document.

          markdown: The Markdown file or Markdown content to extract data from.

          markdown_url: The URL to the Markdown file to extract data from.

          model: The version of the model to use for extraction. Use `extract-latest` to
              use the latest version.

          output_save_url: If zero data retention (ZDR) is enabled, you must enter a URL for the
              extracted output to be saved to. When ZDR is enabled, the extracted content
              will not be in the API response.

          strict: If True, reject schemas with unsupported fields (HTTP 422). If False, prune
              unsupported fields and continue. Only applies to extract versions that
              support schema validation.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_with_paths(
            {
                "schema": schema,
                "markdown": markdown,
                "markdown_url": markdown_url,
                "model": model,
                "output_save_url": output_save_url,
                "strict": strict,
            },
            [["markdown"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return await self._post(
            "/v1/ade/extract/jobs",
            body=await async_maybe_transform(body, extract_job_create_params.ExtractJobCreateParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExtractJobCreateResponse,
        )

    async def get(
        self,
        job_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractJobGetResponse:
        """
        Retrieve the status and results of an async extract job.

        Args:
          job_id: The unique identifier of the extract job.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._get(
            path_template("/v1/ade/extract/jobs/{job_id}", job_id=job_id),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ExtractJobGetResponse,
        )

    async def list(
        self,
        *,
        ending_before: Optional[str] | NotGiven = not_given,
        limit: int | NotGiven = not_given,
        starting_after: Optional[str] | NotGiven = not_given,
        status: Optional[Literal["pending", "processing", "completed", "failed", "cancelled"]] | NotGiven = not_given,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ExtractJobListResponse:
        """
        List async extract jobs with optional filters.

        Args:
          ending_before: A cursor to retrieve results before a specific job ID.

          limit: Number of results to return (default 10, max 100).

          starting_after: A cursor to retrieve results after a specific job ID.

          status: Filter by job status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._get(
            "/v1/ade/extract/jobs",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "ending_before": ending_before,
                        "limit": limit,
                        "starting_after": starting_after,
                        "status": status,
                    },
                    extract_job_list_params.ExtractJobListParams,
                ),
            ),
            cast_to=ExtractJobListResponse,
        )


class ExtractJobsResourceWithRawResponse:
    def __init__(self, extract_jobs: ExtractJobsResource) -> None:
        self._extract_jobs = extract_jobs

        self.create = to_raw_response_wrapper(
            extract_jobs.create,
        )
        self.get = to_raw_response_wrapper(
            extract_jobs.get,
        )
        self.list = to_raw_response_wrapper(
            extract_jobs.list,
        )


class AsyncExtractJobsResourceWithRawResponse:
    def __init__(self, extract_jobs: AsyncExtractJobsResource) -> None:
        self._extract_jobs = extract_jobs

        self.create = async_to_raw_response_wrapper(
            extract_jobs.create,
        )
        self.get = async_to_raw_response_wrapper(
            extract_jobs.get,
        )
        self.list = async_to_raw_response_wrapper(
            extract_jobs.list,
        )


class ExtractJobsResourceWithStreamingResponse:
    def __init__(self, extract_jobs: ExtractJobsResource) -> None:
        self._extract_jobs = extract_jobs

        self.create = to_streamed_response_wrapper(
            extract_jobs.create,
        )
        self.get = to_streamed_response_wrapper(
            extract_jobs.get,
        )
        self.list = to_streamed_response_wrapper(
            extract_jobs.list,
        )


class AsyncExtractJobsResourceWithStreamingResponse:
    def __init__(self, extract_jobs: AsyncExtractJobsResource) -> None:
        self._extract_jobs = extract_jobs

        self.create = async_to_streamed_response_wrapper(
            extract_jobs.create,
        )
        self.get = async_to_streamed_response_wrapper(
            extract_jobs.get,
        )
        self.list = async_to_streamed_response_wrapper(
            extract_jobs.list,
        )