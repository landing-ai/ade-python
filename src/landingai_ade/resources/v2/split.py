# src/landingai_ade/resources/v2/split.py
from __future__ import annotations

import json
from typing import Any, Mapping, Iterable, Optional, cast
from pathlib import Path

import httpx

from ._base import V2ResourceMixin
from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given, extract_files
from ...types.v2 import V2SplitResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._base_client import make_request_options

__all__ = ["SplitResource", "AsyncSplitResource"]


def _encode_split_class(split_class: Iterable[Mapping[str, object]]) -> str:
    # `split_class` rides as a JSON-encoded string form field: an array of objects
    # with `name` (required), `description`, and `identifier` keys.
    return json.dumps([dict(entry) for entry in split_class])


def _build_split_body(
    split_class: Iterable[Mapping[str, object]],
    markdown: object,
    markdown_url: object,
    model: object,
) -> dict[str, Any]:
    raw_body: dict[str, Any] = {
        "split_class": _encode_split_class(split_class),
        "markdown": markdown,
        "markdown_url": markdown_url,
        "model": model,
    }
    # Multipart requests aren't run through `maybe_transform`; drop the
    # `omit`/`not_given` sentinels (and explicit `None`) so unset fields aren't
    # serialized as form fields.
    return {key: value for key, value in raw_body.items() if is_given(value) and value is not None}


class SplitResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        split_class: Iterable[Mapping[str, object]],
        markdown: Optional[FileTypes] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2SplitResponse:
        """Split a Markdown document into segments synchronously against the V2 (ADE)
        `/v2/split` endpoint.

        Consecutive pages with the same classification merge into one segment; an
        identifier change starts a new segment. `/v2/split` is synchronous-only
        (no async jobs route).

        Args:
          split_class: The split classification entries. Each entry is a mapping with a `name`
              (required), and optional `description` and `identifier` keys. At most 19
              entries.

          markdown: The Markdown content to split, as an inline string or a file upload. Either
              this parameter or `markdown_url` must be provided.

          markdown_url: A publicly accessible URL to the Markdown file to split. Either this
              parameter or `markdown` must be provided.

          model: The split model version to use. Defaults to the latest snapshot.

          save_to: Optional output path. If a directory, auto-generates the filename
              (e.g. {input_file}_split_output.json, or split_output.json when no
              input filename is available). If a full path ending in .json, saves there
              directly. Parent directories are created automatically.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        original_markdown, original_markdown_url = markdown, markdown_url
        body = deepcopy_with_paths(
            _build_split_body(split_class, markdown, markdown_url, model),
            [["markdown"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        result = self._post(
            self._v2_url("/v2/split"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2SplitResponse,
        )
        if save_to:
            from ..._client import _save_response, _get_input_filename

            filename = _get_input_filename(original_markdown, original_markdown_url)
            _save_response(save_to, filename, "split", result)
        return result


class AsyncSplitResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        split_class: Iterable[Mapping[str, object]],
        markdown: Optional[FileTypes] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2SplitResponse:
        """Async mirror of `SplitResource.run`. See there for full documentation."""
        original_markdown, original_markdown_url = markdown, markdown_url
        body = deepcopy_with_paths(
            _build_split_body(split_class, markdown, markdown_url, model),
            [["markdown"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["markdown"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        result = await self._post(
            self._v2_url("/v2/split"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2SplitResponse,
        )
        if save_to:
            from ..._client import _save_response, _get_input_filename

            filename = _get_input_filename(original_markdown, original_markdown_url)
            _save_response(save_to, filename, "split", result)
        return result
