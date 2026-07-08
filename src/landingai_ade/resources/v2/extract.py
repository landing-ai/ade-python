from __future__ import annotations

from typing import Any, Dict, Type, Union, Mapping, Optional
from pathlib import Path

import httpx
from pydantic import BaseModel

from ._base import V2ResourceMixin
from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ...types.v2 import V2ExtractResult
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.v2_errors import raise_if_sync_timeout
from ...lib.schema_utils import coerce_schema_to_dict

__all__ = ["ExtractResource", "AsyncExtractResource"]


def _build_extract_body(
    schema: Union[str, Mapping[str, object], Type[BaseModel]],
    markdown: object,
    markdown_ref: object,
    markdown_url: object,
    model: object,
    strict: object,
    idempotency_key: object,
    priority: object = omit,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"schema": coerce_schema_to_dict(schema)}
    for key, value in (
        ("markdown", markdown),
        ("markdown_ref", markdown_ref),
        ("markdown_url", markdown_url),
        ("model", model),
        ("idempotency_key", idempotency_key),
        ("priority", priority),
    ):
        if value is not omit and value is not None:
            body[key] = value
    if strict is not omit and strict is not None:
        body["options"] = {"strict": bool(strict)}
    return body


class ExtractResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        """Extract structured data from markdown synchronously against the V2 (ADE)
        `/v2/extract` endpoint (JSON body).

        Raises `V2SyncTimeoutError` when the server times out the synchronous
        request (HTTP 504); use the async jobs route for long-running documents
        in that case.

        Args:
          schema: JSON schema for field extraction. Accepts a pydantic `BaseModel`
              subclass, a dict, or a JSON-encoded string; it is coerced to a JSON
              object and sent as `schema` in the request body.

          markdown: Markdown content to extract data from.

          markdown_ref: A reference (e.g. from a prior parse) to markdown content to
              extract data from.

          markdown_url: The URL to the markdown file to extract data from.

          model: The version of the model to use for extraction.

          strict: If True, reject schemas with unsupported fields (HTTP 422). If
              False, prune unsupported fields and continue. Sent as
              `options.strict`.

          idempotency_key: An idempotency key for the request.

          save_to: Optional output path. If a directory, auto-generates the filename
              (e.g. {input_file}_extract_output.json, or extract_output.json when no
              input filename is available). If a full path ending in .json, saves there
              directly. Parent directories are created automatically.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_extract_body(schema, markdown, markdown_ref, markdown_url, model, strict, idempotency_key)
        try:
            result = self._post(
                self._v2_url("/v2/extract"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ExtractResult,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _save_response, _get_input_filename

            filename = _get_input_filename(None, markdown_url if isinstance(markdown_url, str) else None)
            _save_response(save_to, filename, "extract", result)
        return result


class AsyncExtractResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        """Async mirror of `ExtractResource.run`. See there for full documentation."""
        body = _build_extract_body(schema, markdown, markdown_ref, markdown_url, model, strict, idempotency_key)
        try:
            result = await self._post(
                self._v2_url("/v2/extract"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ExtractResult,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise
        if save_to:
            from ..._client import _save_response, _get_input_filename

            filename = _get_input_filename(None, markdown_url if isinstance(markdown_url, str) else None)
            _save_response(save_to, filename, "extract", result)
        return result
