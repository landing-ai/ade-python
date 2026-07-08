# src/landingai_ade/resources/v2/parse.py
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, cast
from pathlib import Path

import httpx

from ._base import V2ResourceMixin
from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given, extract_files
from ...types.v2 import V2ParseResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.url_utils import convert_url_to_file_if_local
from ...lib.v2_errors import raise_if_sync_timeout

__all__ = ["ParseResource", "AsyncParseResource"]


def _build_parse_body(
    document: object,
    document_url: object,
    model: object,
    options: object,
    password: object,
) -> dict[str, Any]:
    # `options` is a JSON-encoded string form field per the contract.
    if is_given(options):
        options = json.dumps(options) if not isinstance(options, str) else options
    raw_body = {
        "document": document,
        "document_url": document_url,
        "model": model,
        "options": options,
        "password": password,
    }
    # Multipart requests aren't run through `maybe_transform`, which is what
    # normally strips `omit`/`not_given` sentinels from a params TypedDict --
    # drop them here so unset fields aren't serialized as form fields.
    return {key: value for key, value in raw_body.items() if is_given(value)}


class ParseResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ParseResponse:
        """Parse a document synchronously against the V2 (ADE) `/v2/parse` endpoint.

        Returns a `V2ParseResponse` on both a full success (HTTP 200) and a partial
        success (HTTP 206) -- in the 206 case `result.metadata.failed_pages` lists the
        pages that could not be parsed. Raises `V2SyncTimeoutError` when the server
        times out the synchronous request (HTTP 504); use the async jobs route for
        long-running documents in that case.

        Args:
          document: A file to be parsed. Either this parameter or `document_url` must be provided.

          document_url: The URL to the file to be parsed. Either this parameter or `document` must be
              provided. Local file paths are automatically converted to a `document` upload.

          model: The version of the model to use for parsing.

          options: Additional parsing options. Sent to the server as a JSON-encoded string form
              field.

          password: Password for encrypted document files.

          save_to: Optional output path. If a directory, auto-generates the filename
              (e.g. {input_file}_parse_output.json, or parse_output.json when no
              input filename is available). If a full path ending in .json, saves there
              directly. Parent directories are created automatically.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        original_document, original_document_url = document, document_url
        document, document_url = convert_url_to_file_if_local(document, document_url)
        body = deepcopy_with_paths(
            _build_parse_body(document, document_url, model, options, password),
            [["document"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        try:
            result = self._post(
                self._v2_url("/v2/parse"),
                body=body,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ParseResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _save_response, _get_input_filename

            filename = _get_input_filename(original_document, original_document_url)
            _save_response(save_to, filename, "parse", result)
        return result


class AsyncParseResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ParseResponse:
        """Async mirror of `ParseResource.run`. See there for full documentation."""
        original_document, original_document_url = document, document_url
        document, document_url = convert_url_to_file_if_local(document, document_url)
        body = deepcopy_with_paths(
            _build_parse_body(document, document_url, model, options, password),
            [["document"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        try:
            result = await self._post(
                self._v2_url("/v2/parse"),
                body=body,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ParseResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _save_response, _get_input_filename

            filename = _get_input_filename(original_document, original_document_url)
            _save_response(save_to, filename, "parse", result)
        return result
