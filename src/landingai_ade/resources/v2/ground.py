from __future__ import annotations

import time
from typing import Any, Dict, Union, Mapping, Callable, Optional, cast

import httpx
from pydantic import BaseModel

from ._base import DEFAULT_WAIT_TIMEOUT, JobList, V2ResourceMixin, poll_until_terminal, apoll_until_terminal
from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ..._utils import is_given
from ..._compat import model_dump
from ...types.v2 import Job, V2GroundResult
from ._normalize import normalize_ground_job
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.v2_errors import raise_if_sync_timeout

__all__ = ["GroundResource", "AsyncGroundResource", "GroundJobsResource", "AsyncGroundJobsResource"]


def _as_object(value: Union[Mapping[str, object], BaseModel]) -> Dict[str, Any]:
    """Coerce a mapping or a pydantic model (e.g. a `V2ParseResponse.structure`) to a plain dict."""
    if isinstance(value, BaseModel):
        return model_dump(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected a mapping or pydantic model, received {type(value)!r}")


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

        Raises `V2SyncTimeoutError` when the server times out the synchronous
        request (HTTP 504); use the async jobs route for long-running inputs in
        that case.

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
        try:
            return self._post(
                self._v2_url("/v2/ground"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2GroundResult,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise


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
        try:
            return await self._post(
                self._v2_url("/v2/ground"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2GroundResult,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise


class GroundJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        extraction_metadata: Union[Mapping[str, object], BaseModel],
        structure: Union[Mapping[str, object], BaseModel],
        output_save_url: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Create an asynchronous ground job against `/v2/ground/jobs`.

        Returns a normalized `Job` immediately (typically `pending`). Poll for
        completion via `.get(job_id)`, or block until the job is terminal with
        `.wait(job_id)`.

        Args:
          extraction_metadata: The `extraction_metadata` tree returned by
              `POST /v2/extract`. Accepts a dict or a pydantic model.

          structure: The `structure` tree from the parse response the extraction was
              produced from. Accepts a dict or a pydantic model.

          output_save_url: URL the result should be saved to (e.g. a presigned S3 PUT
              URL) instead of being returned inline. When set, the completed job reports
              `output_url` instead of an inline `result`.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_ground_body(extraction_metadata, structure, output_save_url)
        raw = self._post(
            self._v2_url("/v2/ground/jobs"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_ground_job(cast(Mapping[str, Any], raw))

    def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Get the current status of an async ground job by `job_id`."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v2/ground/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_ground_job(cast(Mapping[str, Any], raw))

    def list(
        self,
        *,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        status: Optional[str] | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> JobList:
        """List async ground jobs associated with your API key, newest first."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = self._get(
            self._v2_url("/v2/ground/jobs"),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=query,
            ),
            cast_to=cast("type[Any]", object),
        )
        env = cast(Mapping[str, Any], raw)
        jobs = [normalize_ground_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
        return JobList.build(jobs, has_more=env.get("has_more"), page=env.get("page"), page_size=env.get("page_size"))

    def wait(
        self,
        job_id: str,
        *,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: Optional[float] = None,
        raise_on_failure: bool = False,
        _monotonic: Optional[Callable[[], float]] = None,
    ) -> Job:
        """Block, polling `.get(job_id)` with backoff, until the job is terminal.

        Raises `JobWaitTimeoutError` if `timeout` seconds elapse before the job
        reaches a terminal state, and `JobFailedError` if `raise_on_failure` is
        set and the job ends failed with an error attached. Ground jobs have no
        `cancelled` status.

        `_monotonic` is a test seam for injecting a fake clock; production
        callers should leave it unset (defaults to `time.monotonic`).
        """
        return poll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            sleep=self._sleep,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )


class AsyncGroundJobsResource(V2ResourceMixin, AsyncAPIResource):
    async def create(
        self,
        *,
        extraction_metadata: Union[Mapping[str, object], BaseModel],
        structure: Union[Mapping[str, object], BaseModel],
        output_save_url: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `GroundJobsResource.create`. See there for full documentation."""
        body = _build_ground_body(extraction_metadata, structure, output_save_url)
        raw = await self._post(
            self._v2_url("/v2/ground/jobs"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_ground_job(cast(Mapping[str, Any], raw))

    async def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `GroundJobsResource.get`. See there for full documentation."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = await self._get(
            self._v2_url(f"/v2/ground/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_ground_job(cast(Mapping[str, Any], raw))

    async def list(
        self,
        *,
        page: int | Omit = omit,
        page_size: int | Omit = omit,
        status: Optional[str] | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> JobList:
        """Async mirror of `GroundJobsResource.list`. See there for full documentation."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = await self._get(
            self._v2_url("/v2/ground/jobs"),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=query,
            ),
            cast_to=cast("type[Any]", object),
        )
        env = cast(Mapping[str, Any], raw)
        jobs = [normalize_ground_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
        return JobList.build(jobs, has_more=env.get("has_more"), page=env.get("page"), page_size=env.get("page_size"))

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: Optional[float] = None,
        raise_on_failure: bool = False,
        _monotonic: Optional[Callable[[], float]] = None,
    ) -> Job:
        """Async mirror of `GroundJobsResource.wait`; sleeps via `anyio.sleep` instead of blocking."""
        return await apoll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )
