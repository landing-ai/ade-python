from __future__ import annotations

from typing import Any, Dict, Union, Mapping

import httpx
from pydantic import BaseModel

from ._base import V2ResourceMixin
from ..._types import Body, Query, Headers, NotGiven, omit, not_given
from ..._utils import is_given
from ..._compat import model_dump
from ...types.v2 import V2GroundResult
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._base_client import make_request_options

__all__ = ["GroundResource", "AsyncGroundResource"]


def _as_object(value: Union[Mapping[str, object], BaseModel]) -> Dict[str, Any]:
    """Coerce a mapping or a pydantic model (e.g. a `V2ParseResponse.structure`) to a plain dict."""
    if isinstance(value, BaseModel):
        return model_dump(value)
    return dict(value)


def _build_ground_body(
    extraction_metadata: Union[Mapping[str, object], BaseModel],
    structure: Union[Mapping[str, object], BaseModel],
    output_save_url: object = omit,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "extraction_metadata": _as_object(extraction_metadata),
        "structure": _as_object(structure),
    }
    if is_given(output_save_url) and output_save_url is not None:
        body["output_save_url"] = output_save_url
    return body


class GroundResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        extraction_metadata: Union[Mapping[str, object], BaseModel],
        structure: Union[Mapping[str, object], BaseModel],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2GroundResult:
        """Map extracted fields to the document blocks they were quoted from against
        the V2 (ADE) `/v2/ground` endpoint (JSON body).

        A pure, stateless join: each `extraction_metadata` leaf's `ranges` are
        overlapped against the `grounding.range` carried on every `structure`
        block, and the matching blocks are returned. Block ids in the response
        resolve only against the `structure` tree supplied here, so pairing an
        extraction with the parse result it actually came from is the caller's
        responsibility.

        Args:
          extraction_metadata: The `extraction_metadata` tree returned by
              `POST /v2/extract` (or `client.v2.extract(...).extraction_metadata`).
              Accepts a dict or a pydantic model.

          structure: The `structure` tree from the parse response the extraction was
              produced from (e.g. `client.v2.parse(...).structure`). Accepts a dict or
              a pydantic model.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_ground_body(extraction_metadata, structure)
        return self._post(
            self._v2_url("/v2/ground"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2GroundResult,
        )


class AsyncGroundResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        extraction_metadata: Union[Mapping[str, object], BaseModel],
        structure: Union[Mapping[str, object], BaseModel],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2GroundResult:
        """Async mirror of `GroundResource.run`. See there for full documentation."""
        body = _build_ground_body(extraction_metadata, structure)
        return await self._post(
            self._v2_url("/v2/ground"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2GroundResult,
        )
