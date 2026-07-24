from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Type, Tuple, Union, Mapping, Callable, Optional, Sequence, cast
from typing_extensions import Literal

import httpx
from pydantic import BaseModel

from ._base import DEFAULT_WAIT_TIMEOUT, JobList, V2ResourceMixin, poll_until_terminal, apoll_until_terminal
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given
from ...types.v2 import Job, V2BuildSchemaResponse
from ._normalize import normalize_build_schema_job
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.v2_errors import raise_if_sync_timeout
from ...lib.schema_utils import coerce_schema_to_dict

__all__ = [
    "BuildSchemaResource",
    "AsyncBuildSchemaResource",
    "BuildSchemaJobsResource",
    "AsyncBuildSchemaJobsResource",
]


def _prepare_build_schema(
    markdowns: object,
    markdown_urls: object,
    prompt: object,
    schema: object,
    service_tier: object = omit,
) -> Tuple[Optional[List[FileTypes]], Optional[List[str]], Optional[str], Optional[str], Optional[str], bool]:
    """Normalize the build-schema inputs and decide the wire encoding.

    Returns ``(markdowns, markdown_urls, prompt, schema, service_tier, use_multipart)``.
    ``schema`` (a pydantic model / dict / JSON string) is coerced to a JSON-encoded
    string, matching the contract's string-typed ``schema`` field. The request is
    sent as ``multipart/form-data`` when any ``markdowns`` entry is a file upload
    (bytes / path / file object / tuple), and as JSON otherwise.
    """
    md_list: Optional[List[FileTypes]] = None
    if is_given(markdowns) and markdowns is not None:
        md_list = [markdowns] if isinstance(markdowns, str) else list(cast(Sequence[FileTypes], markdowns))

    urls_list: Optional[List[str]] = None
    if is_given(markdown_urls) and markdown_urls is not None:
        urls_list = list(cast(Sequence[str], markdown_urls))

    prompt_val: Optional[str] = cast(str, prompt) if (is_given(prompt) and prompt is not None) else None

    schema_val: Optional[str] = None
    if is_given(schema) and schema is not None:
        schema_val = json.dumps(
            coerce_schema_to_dict(cast("Union[str, Mapping[str, object], Type[BaseModel]]", schema))
        )

    tier_val: Optional[str] = cast(str, service_tier) if (is_given(service_tier) and service_tier is not None) else None

    if md_list is None and urls_list is None and prompt_val is None and schema_val is None:
        raise ValueError("build_schema requires at least one of `markdowns`, `markdown_urls`, `prompt`, or `schema`.")

    # An inline markdown is a `str`; anything else (bytes / path / file object /
    # tuple) is a file upload and forces multipart.
    use_multipart = any(not isinstance(m, str) for m in (md_list or []))
    return md_list, urls_list, prompt_val, schema_val, tier_val, use_multipart


def _build_schema_json_body(
    md_list: Optional[List[FileTypes]],
    urls_list: Optional[List[str]],
    prompt_val: Optional[str],
    schema_val: Optional[str],
    tier_val: Optional[str],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if md_list is not None:
        body["markdowns"] = md_list
    if urls_list is not None:
        body["markdown_urls"] = urls_list
    if prompt_val is not None:
        body["prompt"] = prompt_val
    if schema_val is not None:
        body["schema"] = schema_val
    if tier_val is not None:
        body["service_tier"] = tier_val
    return body


def _build_schema_multipart(
    md_list: Optional[List[FileTypes]],
    urls_list: Optional[List[str]],
    prompt_val: Optional[str],
    schema_val: Optional[str],
    tier_val: Optional[str],
) -> Tuple[Dict[str, Any], List[Tuple[str, FileTypes]]]:
    # In multipart form data the list-valued fields are JSON-serialized strings,
    # and every markdown (inline string or file) is a repeated ``markdowns`` part.
    data: Dict[str, Any] = {}
    if urls_list is not None:
        data["markdown_urls"] = json.dumps(urls_list)
    if prompt_val is not None:
        data["prompt"] = prompt_val
    if schema_val is not None:
        data["schema"] = schema_val
    if tier_val is not None:
        data["service_tier"] = tier_val
    files: List[Tuple[str, FileTypes]] = [("markdowns", m) for m in (md_list or [])]
    return data, files


class BuildSchemaResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        markdowns: Optional[Sequence[FileTypes]] | Omit = omit,
        markdown_urls: Optional[Sequence[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Union[str, Mapping[str, object], Type[BaseModel], None] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2BuildSchemaResponse:
        """Generate or edit a JSON Schema for extraction synchronously against
        `/v2/extract/build-schema`.

        At least one of `markdowns`, `markdown_urls`, `prompt`, or `schema` must be
        provided. Raises `V2SyncTimeoutError` when the server times out the
        synchronous request (HTTP 504); use the async jobs route for long-running
        inputs in that case.

        Args:
          markdowns: Markdown files or inline content strings to analyze for schema
              generation. Each entry is either an inline markdown string or a file
              upload (`Path`/`bytes`/file object). Multiple documents can be provided
              for better schema coverage.

          markdown_urls: URLs to Markdown files to analyze for schema generation.

          prompt: Instructions for how to generate or modify the schema.

          schema: Existing JSON schema to iterate on or refine. Accepts a pydantic
              `BaseModel` subclass, a dict, or a JSON-encoded string; it is coerced to
              a JSON Schema and sent as a JSON-encoded string.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        md_list, urls_list, prompt_val, schema_val, tier_val, use_multipart = _prepare_build_schema(
            markdowns, markdown_urls, prompt, schema
        )
        try:
            if use_multipart:
                data, files = _build_schema_multipart(md_list, urls_list, prompt_val, schema_val, tier_val)
                extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
                return self._post(
                    self._v2_url("/v2/extract/build-schema"),
                    body=data,
                    files=files,
                    options=make_request_options(
                        extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                    ),
                    cast_to=V2BuildSchemaResponse,
                )
            body = _build_schema_json_body(md_list, urls_list, prompt_val, schema_val, tier_val)
            return self._post(
                self._v2_url("/v2/extract/build-schema"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2BuildSchemaResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise


class AsyncBuildSchemaResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        markdowns: Optional[Sequence[FileTypes]] | Omit = omit,
        markdown_urls: Optional[Sequence[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Union[str, Mapping[str, object], Type[BaseModel], None] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2BuildSchemaResponse:
        """Async mirror of `BuildSchemaResource.run`. See there for full documentation."""
        md_list, urls_list, prompt_val, schema_val, tier_val, use_multipart = _prepare_build_schema(
            markdowns, markdown_urls, prompt, schema
        )
        try:
            if use_multipart:
                data, files = _build_schema_multipart(md_list, urls_list, prompt_val, schema_val, tier_val)
                extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
                return await self._post(
                    self._v2_url("/v2/extract/build-schema"),
                    body=data,
                    files=files,
                    options=make_request_options(
                        extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                    ),
                    cast_to=V2BuildSchemaResponse,
                )
            body = _build_schema_json_body(md_list, urls_list, prompt_val, schema_val, tier_val)
            return await self._post(
                self._v2_url("/v2/extract/build-schema"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2BuildSchemaResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc)
            raise


class BuildSchemaJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        markdowns: Optional[Sequence[FileTypes]] | Omit = omit,
        markdown_urls: Optional[Sequence[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Union[str, Mapping[str, object], Type[BaseModel], None] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Create an asynchronous build-schema job against `/v2/extract/build-schema/jobs`.

        Returns a normalized `Job` immediately (typically `pending`). Poll for
        completion via `.get(job_id)`, or block until the job is terminal with
        `.wait(job_id)`.

        Args:
          markdowns: Markdown files or inline content strings to analyze for schema
              generation. Each entry is either an inline markdown string or a file
              upload (`Path`/`bytes`/file object).

          markdown_urls: URLs to Markdown files to analyze for schema generation.

          prompt: Instructions for how to generate or modify the schema.

          schema: Existing JSON schema to iterate on or refine. Accepts a pydantic
              `BaseModel` subclass, a dict, or a JSON-encoded string.

          service_tier: Service tier for the job: ``standard`` or ``priority``.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        md_list, urls_list, prompt_val, schema_val, tier_val, use_multipart = _prepare_build_schema(
            markdowns, markdown_urls, prompt, schema, service_tier
        )
        if use_multipart:
            data, files = _build_schema_multipart(md_list, urls_list, prompt_val, schema_val, tier_val)
            extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
            raw = self._post(
                self._v2_url("/v2/extract/build-schema/jobs"),
                body=data,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=cast("type[Any]", object),
            )
        else:
            body = _build_schema_json_body(md_list, urls_list, prompt_val, schema_val, tier_val)
            raw = self._post(
                self._v2_url("/v2/extract/build-schema/jobs"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=cast("type[Any]", object),
            )
        return normalize_build_schema_job(cast(Mapping[str, Any], raw))

    def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Get the current status of an async build-schema job by `job_id`."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v2/extract/build-schema/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_build_schema_job(cast(Mapping[str, Any], raw))

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
        """List async build-schema jobs associated with your API key, newest first."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = self._get(
            self._v2_url("/v2/extract/build-schema/jobs"),
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
        jobs = [normalize_build_schema_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
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
        set and the job ends failed with an error attached. Build-schema jobs have
        no `cancelled` status.

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


class AsyncBuildSchemaJobsResource(V2ResourceMixin, AsyncAPIResource):
    async def create(
        self,
        *,
        markdowns: Optional[Sequence[FileTypes]] | Omit = omit,
        markdown_urls: Optional[Sequence[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Union[str, Mapping[str, object], Type[BaseModel], None] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `BuildSchemaJobsResource.create`. See there for full documentation."""
        md_list, urls_list, prompt_val, schema_val, tier_val, use_multipart = _prepare_build_schema(
            markdowns, markdown_urls, prompt, schema, service_tier
        )
        if use_multipart:
            data, files = _build_schema_multipart(md_list, urls_list, prompt_val, schema_val, tier_val)
            extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
            raw = await self._post(
                self._v2_url("/v2/extract/build-schema/jobs"),
                body=data,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=cast("type[Any]", object),
            )
        else:
            body = _build_schema_json_body(md_list, urls_list, prompt_val, schema_val, tier_val)
            raw = await self._post(
                self._v2_url("/v2/extract/build-schema/jobs"),
                body=body,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=cast("type[Any]", object),
            )
        return normalize_build_schema_job(cast(Mapping[str, Any], raw))

    async def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `BuildSchemaJobsResource.get`. See there for full documentation."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = await self._get(
            self._v2_url(f"/v2/extract/build-schema/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_build_schema_job(cast(Mapping[str, Any], raw))

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
        """Async mirror of `BuildSchemaJobsResource.list`. See there for full documentation."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = await self._get(
            self._v2_url("/v2/extract/build-schema/jobs"),
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
        jobs = [normalize_build_schema_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
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
        """Async mirror of `BuildSchemaJobsResource.wait`; sleeps via `anyio.sleep` instead of blocking."""
        return await apoll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )
