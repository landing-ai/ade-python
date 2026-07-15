from __future__ import annotations

import time
from typing import Any, Dict, Type, Union, Mapping, Callable, Optional, cast
from pathlib import Path
from typing_extensions import Literal

import httpx
from pydantic import BaseModel

from ._base import DEFAULT_WAIT_TIMEOUT, JobList, V2ResourceMixin, poll_until_terminal, apoll_until_terminal
from ..._types import Body, Omit, Query, Headers, NotGiven, omit, not_given
from ..._utils import is_given
from ...types.v2 import Job, V2ExtractResult
from ._normalize import normalize_extract_job
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.v2_errors import raise_if_sync_timeout
from ...lib.schema_utils import coerce_schema_to_dict

__all__ = ["ExtractResource", "AsyncExtractResource", "ExtractJobsResource", "AsyncExtractJobsResource"]


def _build_extract_body(
    schema: Union[str, Mapping[str, object], Type[BaseModel]],
    markdown: object,
    markdown_url: object,
    model: object,
    strict: object,
    service_tier: object = omit,
) -> Dict[str, Any]:
    provided = [
        name
        for name, value in (
            ("markdown", markdown),
            ("markdown_url", markdown_url),
        )
        if value is not omit and value is not None
    ]
    if len(provided) != 1:
        raise ValueError(
            "extract requires exactly one markdown source: provide one of "
            "`markdown` or `markdown_url`" + (f" (received: {', '.join(provided)})" if provided else "") + "."
        )
    body: Dict[str, Any] = {"schema": coerce_schema_to_dict(schema)}
    for key, value in (
        ("markdown", markdown),
        ("markdown_url", markdown_url),
        ("model", model),
        ("service_tier", service_tier),
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
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
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

          markdown_url: The URL to the markdown file to extract data from.

          model: The version of the model to use for extraction.

          strict: If True, reject schemas with unsupported fields (HTTP 422). If
              False, prune unsupported fields and continue. Sent as
              `options.strict`.

          save_to: Optional output path. If a directory, auto-generates the filename
              (e.g. {input_file}_extract_output.json, or extract_output.json when no
              input filename is available). If a full path ending in .json, saves there
              directly. Parent directories are created automatically.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_extract_body(schema, markdown, markdown_url, model, strict)
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
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        """Async mirror of `ExtractResource.run`. See there for full documentation."""
        body = _build_extract_body(schema, markdown, markdown_url, model, strict)
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


class ExtractJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Create an asynchronous extract job against `/v2/extract/jobs`.

        Returns a normalized `Job` immediately (typically `pending`). Poll for
        completion via `.get(job_id)`, or block until the job is terminal with
        `.wait(job_id)`.

        Args:
          schema: JSON schema for field extraction. Accepts a pydantic `BaseModel`
              subclass, a dict, or a JSON-encoded string; it is coerced to a JSON
              object and sent as `schema` in the request body.

          markdown: Markdown content to extract data from.

          markdown_url: The URL to the markdown file to extract data from.

          model: The version of the model to use for extraction.

          strict: If True, reject schemas with unsupported fields (HTTP 422). If
              False, prune unsupported fields and continue. Sent as
              `options.strict`.

          service_tier: Service tier for the job: ``standard`` or ``priority``.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_extract_body(schema, markdown, markdown_url, model, strict, service_tier)
        raw = self._post(
            self._v2_url("/v2/extract/jobs"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_extract_job(cast(Mapping[str, Any], raw))

    def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Get the current status of an async extract job by `job_id`."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v2/extract/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_extract_job(cast(Mapping[str, Any], raw))

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
        """List async extract jobs associated with your API key, newest first."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = self._get(
            self._v2_url("/v2/extract/jobs"),
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
        jobs = [normalize_extract_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
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
        set and the job ends failed with an error attached. Extract jobs have no
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


class AsyncExtractJobsResource(V2ResourceMixin, AsyncAPIResource):
    async def create(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `ExtractJobsResource.create`. See there for full documentation."""
        body = _build_extract_body(schema, markdown, markdown_url, model, strict, service_tier)
        raw = await self._post(
            self._v2_url("/v2/extract/jobs"),
            body=body,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_extract_job(cast(Mapping[str, Any], raw))

    async def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `ExtractJobsResource.get`. See there for full documentation."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = await self._get(
            self._v2_url(f"/v2/extract/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_extract_job(cast(Mapping[str, Any], raw))

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
        """Async mirror of `ExtractJobsResource.list`. See there for full documentation."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = await self._get(
            self._v2_url("/v2/extract/jobs"),
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
        jobs = [normalize_extract_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
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
        """Async mirror of `ExtractJobsResource.wait`; sleeps via `anyio.sleep` instead of blocking."""
        return await apoll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )
