# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Mapping, Optional, cast
from typing_extensions import Literal

import httpx

from ..types import parse_job_list_params, parse_job_create_params
from .._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from .._utils import extract_files, maybe_transform, deepcopy_minimal, async_maybe_transform
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.parse_job_get_response import ParseJobGetResponse
from ..types.parse_job_list_response import ParseJobListResponse
from ..types.parse_job_create_response import ParseJobCreateResponse

__all__ = ["ParseJobsResource", "AsyncParseJobsResource"]


class ParseJobsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ParseJobsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/landing-ai/ade-python#accessing-raw-response-data-eg-headers
        """
        return ParseJobsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ParseJobsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/landing-ai/ade-python#with_streaming_response
        """
        return ParseJobsResourceWithStreamingResponse(self)

    def create(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        split: Optional[Literal["page"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ParseJobCreateResponse:
        """
        Parse documents asynchronously.

        This endpoint creates a job that handles the processing for both large documents
        and large batches of documents.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/parse/jobs`.

        Args:
          document: A file to be parsed. The file can be a PDF or an image. See the list of
              supported file types here: https://docs.landing.ai/ade/ade-file-types. Either
              this parameter or the `document_url` parameter must be provided.

          document_url: The URL to the file to be parsed. The file can be a PDF or an image. See the
              list of supported file types here: https://docs.landing.ai/ade/ade-file-types.
              Either this parameter or the `document` parameter must be provided.

          model: The version of the model to use for parsing.

          output_save_url: If zero data retention (ZDR) is enabled, you must enter a URL for the parsed
              output to be saved to. When ZDR is enabled, the parsed content will not be in
              the API response.

          split: If you want to split documents into smaller sections, include the split
              parameter. Set the parameter to page to split documents at the page level. The
              splits object in the API output will contain a set of data for each page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal(
            {
                "document": document,
                "document_url": document_url,
                "model": model,
                "output_save_url": output_save_url,
                "split": split,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            "/v1/ade/parse/jobs",
            body=maybe_transform(body, parse_job_create_params.ParseJobCreateParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ParseJobCreateResponse,
        )

    def list(
        self,
        *,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        status: Optional[Literal["cancelled", "completed", "failed", "pending", "processing"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ParseJobListResponse:
        """List all async parse jobs associated with your API key.

        Returns the list of jobs
        or an error response. For EU users, use this endpoint:

        `https://api.va.eu-west-1.landing.ai/v1/ade/parse/jobs`.

        Args:
          page: Page number (0-indexed)

          page_size: Number of items per page

          status: Filter by job status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get(
            "/v1/ade/parse/jobs",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform(
                    {
                        "page": page,
                        "page_size": page_size,
                        "status": status,
                    },
                    parse_job_list_params.ParseJobListParams,
                ),
            ),
            cast_to=ParseJobListResponse,
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
    ) -> ParseJobGetResponse:
        """
        Get the status for an async parse job.

        Returns the job status or an error response. For EU users, use this endpoint:

        `https://api.va.eu-west-1.landing.ai/v1/ade/parse/jobs/{job_id}`.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        return self._get(
            f"/v1/ade/parse/jobs/{job_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ParseJobGetResponse,
        )


class AsyncParseJobsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncParseJobsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/landing-ai/ade-python#accessing-raw-response-data-eg-headers
        """
        return AsyncParseJobsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncParseJobsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/landing-ai/ade-python#with_streaming_response
        """
        return AsyncParseJobsResourceWithStreamingResponse(self)

    async def create(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        split: Optional[Literal["page"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ParseJobCreateResponse:
        """
        Parse documents asynchronously.

        This endpoint creates a job that handles the processing for both large documents
        and large batches of documents.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/parse/jobs`.

        Args:
          document: A file to be parsed. The file can be a PDF or an image. See the list of
              supported file types here: https://docs.landing.ai/ade/ade-file-types. Either
              this parameter or the `document_url` parameter must be provided.

          document_url: The URL to the file to be parsed. The file can be a PDF or an image. See the
              list of supported file types here: https://docs.landing.ai/ade/ade-file-types.
              Either this parameter or the `document` parameter must be provided.

          model: The version of the model to use for parsing.

          output_save_url: If zero data retention (ZDR) is enabled, you must enter a URL for the parsed
              output to be saved to. When ZDR is enabled, the parsed content will not be in
              the API response.

          split: If you want to split documents into smaller sections, include the split
              parameter. Set the parameter to page to split documents at the page level. The
              splits object in the API output will contain a set of data for each page.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal(
            {
                "document": document,
                "document_url": document_url,
                "model": model,
                "output_save_url": output_save_url,
                "split": split,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return await self._post(
            "/v1/ade/parse/jobs",
            body=await async_maybe_transform(body, parse_job_create_params.ParseJobCreateParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ParseJobCreateResponse,
        )

    async def list(
        self,
        *,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        status: Optional[Literal["cancelled", "completed", "failed", "pending", "processing"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ParseJobListResponse:
        """List all async parse jobs associated with your API key.

        Returns the list of jobs
        or an error response. For EU users, use this endpoint:

        `https://api.va.eu-west-1.landing.ai/v1/ade/parse/jobs`.

        Args:
          page: Page number (0-indexed)

          page_size: Number of items per page

          status: Filter by job status.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._get(
            "/v1/ade/parse/jobs",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=await async_maybe_transform(
                    {
                        "page": page,
                        "page_size": page_size,
                        "status": status,
                    },
                    parse_job_list_params.ParseJobListParams,
                ),
            ),
            cast_to=ParseJobListResponse,
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
    ) -> ParseJobGetResponse:
        """
        Get the status for an async parse job.

        Returns the job status or an error response. For EU users, use this endpoint:

        `https://api.va.eu-west-1.landing.ai/v1/ade/parse/jobs/{job_id}`.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        return await self._get(
            f"/v1/ade/parse/jobs/{job_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ParseJobGetResponse,
        )


class ParseJobsResourceWithRawResponse:
    def __init__(self, parse_jobs: ParseJobsResource) -> None:
        self._parse_jobs = parse_jobs

        self.create = to_raw_response_wrapper(
            parse_jobs.create,
        )
        self.list = to_raw_response_wrapper(
            parse_jobs.list,
        )
        self.get = to_raw_response_wrapper(
            parse_jobs.get,
        )


class AsyncParseJobsResourceWithRawResponse:
    def __init__(self, parse_jobs: AsyncParseJobsResource) -> None:
        self._parse_jobs = parse_jobs

        self.create = async_to_raw_response_wrapper(
            parse_jobs.create,
        )
        self.list = async_to_raw_response_wrapper(
            parse_jobs.list,
        )
        self.get = async_to_raw_response_wrapper(
            parse_jobs.get,
        )


class ParseJobsResourceWithStreamingResponse:
    def __init__(self, parse_jobs: ParseJobsResource) -> None:
        self._parse_jobs = parse_jobs

        self.create = to_streamed_response_wrapper(
            parse_jobs.create,
        )
        self.list = to_streamed_response_wrapper(
            parse_jobs.list,
        )
        self.get = to_streamed_response_wrapper(
            parse_jobs.get,
        )


class AsyncParseJobsResourceWithStreamingResponse:
    def __init__(self, parse_jobs: AsyncParseJobsResource) -> None:
        self._parse_jobs = parse_jobs

        self.create = async_to_streamed_response_wrapper(
            parse_jobs.create,
        )
        self.list = async_to_streamed_response_wrapper(
            parse_jobs.list,
        )
        self.get = async_to_streamed_response_wrapper(
            parse_jobs.get,
        )
