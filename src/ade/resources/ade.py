# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Mapping, Optional, cast
from typing_extensions import Literal

import httpx

from ..types import ade_parse_params, ade_extract_params
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
from ..types.ade_parse_response import AdeParseResponse
from ..types.ade_extract_response import AdeExtractResponse

__all__ = ["AdeResource", "AsyncAdeResource"]


class AdeResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AdeResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/landing-ai/ade-python#accessing-raw-response-data-eg-headers
        """
        return AdeResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AdeResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/landing-ai/ade-python#with_streaming_response
        """
        return AdeResourceWithStreamingResponse(self)

    def extract(
        self,
        *,
        schema: str,
        markdown: Optional[FileTypes] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AdeExtractResponse:
        """
        Extract structured data from Markdown using a JSON schema.

        This endpoint processes Markdown content and extracts structured data according
        to the provided JSON schema.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/extract`.

        Args:
          schema: JSON schema for field extraction. This schema determines what key-values pairs
              are extracted from the Markdown. The schema must be a valid JSON object and will
              be validated before processing the document.

          markdown: The Markdown file to extract data from.

          markdown_url: The URL to the Markdown file to extract data from.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal(
            {
                "schema": schema,
                "markdown": markdown,
                "markdown_url": markdown_url,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            "/v1/ade/extract",
            body=maybe_transform(body, ade_extract_params.AdeExtractParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=AdeExtractResponse,
        )

    def parse(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        split: Optional[Literal["page"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AdeParseResponse:
        """
        Parse a document.

        This endpoint parses documents and structured Markdown, chunks, and metadata.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/parse`.

        Args:
          document: A file to be parsed. The file can be a PDF (50 pages max) or an image (50MB).
              See the list of supported file types here
              (https://docs.landing.ai/ade/ade-file-types). Either this parameter or the
              document_url parameter must be provided.

          document_url: The URL to the file to be parsed. The file can be a PDF (50 pages max) or an
              image (50MB). See the list of supported file types here
              (https://docs.landing.ai/ade/ade-file-types). Either this parameter or the
              document parameter must be provided.

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
                "split": split,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            "/v1/ade/parse",
            body=maybe_transform(body, ade_parse_params.AdeParseParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=AdeParseResponse,
        )


class AsyncAdeResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncAdeResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/landing-ai/ade-python#accessing-raw-response-data-eg-headers
        """
        return AsyncAdeResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAdeResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/landing-ai/ade-python#with_streaming_response
        """
        return AsyncAdeResourceWithStreamingResponse(self)

    async def extract(
        self,
        *,
        schema: str,
        markdown: Optional[FileTypes] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AdeExtractResponse:
        """
        Extract structured data from Markdown using a JSON schema.

        This endpoint processes Markdown content and extracts structured data according
        to the provided JSON schema.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/extract`.

        Args:
          schema: JSON schema for field extraction. This schema determines what key-values pairs
              are extracted from the Markdown. The schema must be a valid JSON object and will
              be validated before processing the document.

          markdown: The Markdown file to extract data from.

          markdown_url: The URL to the Markdown file to extract data from.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal(
            {
                "schema": schema,
                "markdown": markdown,
                "markdown_url": markdown_url,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return await self._post(
            "/v1/ade/extract",
            body=await async_maybe_transform(body, ade_extract_params.AdeExtractParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=AdeExtractResponse,
        )

    async def parse(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        split: Optional[Literal["page"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> AdeParseResponse:
        """
        Parse a document.

        This endpoint parses documents and structured Markdown, chunks, and metadata.

        For EU users, use this endpoint:

            `https://api.va.eu-west-1.landing.ai/v1/ade/parse`.

        Args:
          document: A file to be parsed. The file can be a PDF (50 pages max) or an image (50MB).
              See the list of supported file types here
              (https://docs.landing.ai/ade/ade-file-types). Either this parameter or the
              document_url parameter must be provided.

          document_url: The URL to the file to be parsed. The file can be a PDF (50 pages max) or an
              image (50MB). See the list of supported file types here
              (https://docs.landing.ai/ade/ade-file-types). Either this parameter or the
              document parameter must be provided.

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
                "split": split,
            }
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return await self._post(
            "/v1/ade/parse",
            body=await async_maybe_transform(body, ade_parse_params.AdeParseParams),
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=AdeParseResponse,
        )


class AdeResourceWithRawResponse:
    def __init__(self, ade: AdeResource) -> None:
        self._ade = ade

        self.extract = to_raw_response_wrapper(
            ade.extract,
        )
        self.parse = to_raw_response_wrapper(
            ade.parse,
        )


class AsyncAdeResourceWithRawResponse:
    def __init__(self, ade: AsyncAdeResource) -> None:
        self._ade = ade

        self.extract = async_to_raw_response_wrapper(
            ade.extract,
        )
        self.parse = async_to_raw_response_wrapper(
            ade.parse,
        )


class AdeResourceWithStreamingResponse:
    def __init__(self, ade: AdeResource) -> None:
        self._ade = ade

        self.extract = to_streamed_response_wrapper(
            ade.extract,
        )
        self.parse = to_streamed_response_wrapper(
            ade.parse,
        )


class AsyncAdeResourceWithStreamingResponse:
    def __init__(self, ade: AsyncAdeResource) -> None:
        self._ade = ade

        self.extract = async_to_streamed_response_wrapper(
            ade.extract,
        )
        self.parse = async_to_streamed_response_wrapper(
            ade.parse,
        )
