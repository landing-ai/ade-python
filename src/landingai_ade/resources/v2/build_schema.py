from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Type, Union, Mapping, Callable, Optional, cast
from typing_extensions import Literal

import httpx
from pydantic import BaseModel

from ._base import DEFAULT_WAIT_TIMEOUT, JobList, V2ResourceMixin, poll_until_terminal, apoll_until_terminal
from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given, extract_files
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


def _coerce_schema_string(schema: Union[str, Mapping[str, object], Type[BaseModel]]) -> str:
    """Accept a pydantic model, a dict, or a JSON string; return a JSON-Schema string.

    The build-schema `schema` field is sent as a string (VTRA parity), so the
    coerced dict is re-serialized to JSON.
    """
    return json.dumps(coerce_schema_to_dict(schema))


def _build_build_schema_body(
    markdowns: object,
    markdown_urls: object,
    prompt: object,
    schema: object,
    service_tier: object = omit,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if markdowns is not omit and markdowns is not None:
        body["markdowns"] = list(cast("List[Any]", markdowns))
    if markdown_urls is not omit and markdown_urls is not None:
        body["markdown_urls"] = list(cast(List[str], markdown_urls))
    if prompt is not omit and prompt is not None:
        body["prompt"] = prompt
    if schema is not omit and schema is not None:
        body["schema"] = _coerce_schema_string(cast(Union[str, Mapping[str, object], Type[BaseModel]], schema))
    if not body:
        raise ValueError(
            "build_schema requires at least one source: provide one of "
            "`markdowns`, `markdown_urls`, `prompt`, or `schema`."
        )
    if service_tier is not omit and service_tier is not None:
        body["service_tier"] = service_tier
    return body


# The `markdowns` array is the only file-capable field: the spec's
# `multipart/form-data` variant types each entry as `str | binary`.
_MARKDOWNS_FILE_PATHS = [["markdowns", "<array>"]]


def _markdowns_has_file(markdowns: object) -> bool:
    """True when any `markdowns` entry is a file upload rather than inline text.

    A plain `str` is inline markdown content (JSON-capable); anything else
    (`Path`, `bytes`, a file object, or a `(filename, content)` tuple) is a file
    upload, which forces the multipart request variant. Decided from the value's
    type per the spec's `format: binary` markdowns item -- never from a generated
    model's class name.
    """
    if not is_given(markdowns) or markdowns is None:
        return False
    return any(not isinstance(item, str) for item in cast("List[Any]", markdowns))


def _prepare_build_schema_request(
    body: Dict[str, Any],
    markdowns: object,
    extra_headers: Headers | None,
) -> tuple[Dict[str, Any], Optional[List[tuple[str, FileTypes]]], Headers | None]:
    """Route the request through multipart when `markdowns` carries a file.

    Returns `(body, files, extra_headers)`. When no markdown is a file this is a
    no-op and the caller sends JSON (`files` is `None`); otherwise the markdown
    entries are pulled out as `multipart/form-data` parts, mirroring how
    `parse.py` uploads its `document`.
    """
    if not _markdowns_has_file(markdowns):
        return body, None, extra_headers
    body = deepcopy_with_paths(body, _MARKDOWNS_FILE_PATHS)
    files = extract_files(cast(Mapping[str, object], body), paths=_MARKDOWNS_FILE_PATHS)
    extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
    return body, files, extra_headers


class BuildSchemaResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        markdowns: Optional[List[FileTypes]] | Omit = omit,
        markdown_urls: Optional[List[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Optional[Union[str, Mapping[str, object], Type[BaseModel]]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2BuildSchemaResponse:
        """Generate or refine a JSON Schema for extraction synchronously against the
        V2 (ADE) `/v2/extract/build-schema` endpoint.

        Sends a JSON body, or `multipart/form-data` when any `markdowns` entry is a
        file upload (a `Path`/`bytes`/file object rather than an inline string).
        At least one of `markdowns`, `markdown_urls`, `prompt`, or `schema` must be
        provided. Raises `V2SyncTimeoutError` when the server times out the
        synchronous request (HTTP 504); use the async jobs route for long-running
        inputs in that case.

        Args:
          markdowns: Markdown files or inline content strings to analyze for schema
              generation. Multiple documents can be provided for better coverage.

          markdown_urls: URLs to Markdown files to analyze for schema generation.

          prompt: Natural-language instructions for how to generate or modify the schema.

          schema: An existing JSON schema to iterate on or refine. Accepts a pydantic
              `BaseModel` subclass, a dict, or a JSON-encoded string; it is coerced to a
              JSON Schema and sent as the `schema` string in the request body.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_build_schema_body(markdowns, markdown_urls, prompt, schema)
        body, files, extra_headers = _prepare_build_schema_request(body, markdowns, extra_headers)
        try:
            return self._post(
                self._v2_url("/v2/extract/build-schema"),
                body=body,
                files=files,
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
        markdowns: Optional[List[FileTypes]] | Omit = omit,
        markdown_urls: Optional[List[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Optional[Union[str, Mapping[str, object], Type[BaseModel]]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2BuildSchemaResponse:
        """Async mirror of `BuildSchemaResource.run`. See there for full documentation."""
        body = _build_build_schema_body(markdowns, markdown_urls, prompt, schema)
        body, files, extra_headers = _prepare_build_schema_request(body, markdowns, extra_headers)
        try:
            return await self._post(
                self._v2_url("/v2/extract/build-schema"),
                body=body,
                files=files,
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
        markdowns: Optional[List[FileTypes]] | Omit = omit,
        markdown_urls: Optional[List[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Optional[Union[str, Mapping[str, object], Type[BaseModel]]] | Omit = omit,
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
              generation. Multiple documents can be provided for better coverage.

          markdown_urls: URLs to Markdown files to analyze for schema generation.

          prompt: Natural-language instructions for how to generate or modify the schema.

          schema: An existing JSON schema to iterate on or refine. Accepts a pydantic
              `BaseModel` subclass, a dict, or a JSON-encoded string.

          service_tier: Service tier for the job: ``standard`` or ``priority``.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_build_schema_body(markdowns, markdown_urls, prompt, schema, service_tier)
        body, files, extra_headers = _prepare_build_schema_request(body, markdowns, extra_headers)
        raw = self._post(
            self._v2_url("/v2/extract/build-schema/jobs"),
            body=body,
            files=files,
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
        markdowns: Optional[List[FileTypes]] | Omit = omit,
        markdown_urls: Optional[List[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Optional[Union[str, Mapping[str, object], Type[BaseModel]]] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `BuildSchemaJobsResource.create`. See there for full documentation."""
        body = _build_build_schema_body(markdowns, markdown_urls, prompt, schema, service_tier)
        body, files, extra_headers = _prepare_build_schema_request(body, markdowns, extra_headers)
        raw = await self._post(
            self._v2_url("/v2/extract/build-schema/jobs"),
            body=body,
            files=files,
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
