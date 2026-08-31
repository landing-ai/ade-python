from __future__ import annotations

import json
from typing import Any, Mapping, Iterable, Optional, cast

import httpx

from ._base import V2ResourceMixin
from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given, extract_files
from ...types.v2 import V2SplitResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._base_client import make_request_options

__all__ = ["SplitResource", "AsyncSplitResource"]


def _build_split_body(
    markdown: object,
    markdown_url: object,
    split_class: Iterable[Mapping[str, object]],
    model: object,
) -> dict[str, Any]:
    # `split_class` is a JSON-encoded string form field per the contract: an
    # array of objects with `name` (required), `description`, and `identifier`.
    body: dict[str, Any] = {"split_class": json.dumps([dict(entry) for entry in split_class])}
    # Multipart requests aren't run through `maybe_transform`, so drop unset
    # `omit`/`not_given` sentinels (and explicit `None`) here so they aren't
    # serialized as form fields.
    for key, value in (
        ("markdown", markdown),
        ("markdown_url", markdown_url),
        ("model", model),
    ):
        if is_given(value) and value is not None:
            body[key] = value
    return body


class SplitResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        split_class: Iterable[Mapping[str, object]],
        markdown: Optional[FileTypes] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2SplitResponse:
        """Split a Markdown document into segments synchronously against the V2 (ADE)
        `/v1/split` endpoint (multipart body).

        Consecutive pages with the same classification merge into one segment; an
        identifier change starts a new segment. `/v1/split` is synchronous-only
        (no async jobs route).

        Args:
          split_class: The split classification entries. Each entry is a mapping with a `name`
              (required), and optional `description` / `identifier` keys. At most 19
              entries. Sent to the server as a JSON-encoded string form field.

          markdown: Markdown content to split, as an inline string or an uploaded file. Either
              this parameter or `markdown_url` must be provided.

          markdown_url: The URL to the Markdown file to split. Either this parameter or `markdown`
              must be provided.

          model: The split model version to use (e.g. `split-latest`). The resolved version
              is echoed back as `metadata.version`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_with_paths(
            _build_split_body(markdown, markdown_url, split_class, model),
            [["markdown"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            self._v2_url("/v1/split"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2SplitResponse,
        )


class AsyncSplitResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        split_class: Iterable[Mapping[str, object]],
        markdown: Optional[FileTypes] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2SplitResponse:
        """Async mirror of `SplitResource.run`. See there for full documentation."""
        body = deepcopy_with_paths(
            _build_split_body(markdown, markdown_url, split_class, model),
            [["markdown"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return await self._post(
            self._v2_url("/v1/split"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2SplitResponse,
        )
