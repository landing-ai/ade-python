# src/landingai_ade/resources/v2/parse.py
from __future__ import annotations

import json
import time
from typing import Any, Mapping, Callable, Optional, cast
from pathlib import Path
from typing_extensions import Literal

import httpx

from ._base import DEFAULT_WAIT_TIMEOUT, JobList, V2ResourceMixin, poll_until_terminal, apoll_until_terminal
from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given, extract_files
from ...types.v2 import Job, V2ParseResponse
from ._normalize import normalize_parse_job
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.url_utils import convert_url_to_file_if_local
from ...lib.v2_errors import raise_if_sync_timeout

__all__ = ["ParseResource", "AsyncParseResource", "ParseJobsResource", "AsyncParseJobsResource"]


def _build_parse_body(
    document: object,
    document_url: object,
    model: object,
    options: object,
    password: object,
) -> dict[str, Any]:
    # `options` is a JSON-encoded string form field per the contract.
    if is_given(options) and options is not None:
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
    # drop them here so unset fields aren't serialized as form fields. An
    # explicit `None` is likewise treated as "unset" for these optional wire
    # fields, since `is_given` only filters the `omit`/`not_given` sentinels
    # and otherwise returns True for `None`.
    return {key: value for key, value in raw_body.items() if is_given(value) and value is not None}


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


class ParseJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Create an asynchronous parse job against `/v2/parse/jobs`.

        Returns a normalized `Job` immediately (typically `pending`). Poll for
        completion via `.get(job_id)`, or block until the job is terminal with
        `.wait(job_id)`.

        Args:
          document: A file to be parsed. Either this parameter or `document_url` must be provided.

          document_url: The URL to the file to be parsed. Either this parameter or `document` must be
              provided.

          model: The version of the model to use for parsing.

          options: Additional parsing options. Sent to the server as a JSON-encoded string form
              field.

          password: Password for encrypted document files.

          output_save_url: If zero data retention (ZDR) is enabled, a URL the parsed output should be
              saved to instead of being returned in the job result.

          service_tier: Service tier for the job: ``standard`` or ``priority``.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_parse_body(document, document_url, model, options, password)
        if is_given(output_save_url) and output_save_url is not None:
            body["output_save_url"] = output_save_url
        if is_given(service_tier) and service_tier is not None:
            body["service_tier"] = service_tier
        body = deepcopy_with_paths(body, [["document"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        raw = self._post(
            self._v2_url("/v2/parse/jobs"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_parse_job(cast(Mapping[str, Any], raw))

    def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Get the current status of an async parse job by `job_id`."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v2/parse/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_parse_job(cast(Mapping[str, Any], raw))

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
        """List async parse jobs associated with your API key, newest first."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = self._get(
            self._v2_url("/v2/parse/jobs"),
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
        jobs = [normalize_parse_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
        return JobList.build(
            jobs,
            has_more=env.get("has_more"),
            org_id=env.get("org_id"),
            page=env.get("page"),
            page_size=env.get("page_size"),
        )

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
        set and the job ends failed/cancelled with an error attached.

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


class AsyncParseJobsResource(V2ResourceMixin, AsyncAPIResource):
    async def create(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        output_save_url: Optional[str] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `ParseJobsResource.create`. See there for full documentation."""
        body = _build_parse_body(document, document_url, model, options, password)
        if is_given(output_save_url) and output_save_url is not None:
            body["output_save_url"] = output_save_url
        if is_given(service_tier) and service_tier is not None:
            body["service_tier"] = service_tier
        body = deepcopy_with_paths(body, [["document"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        raw = await self._post(
            self._v2_url("/v2/parse/jobs"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_parse_job(cast(Mapping[str, Any], raw))

    async def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `ParseJobsResource.get`. See there for full documentation."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = await self._get(
            self._v2_url(f"/v2/parse/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_parse_job(cast(Mapping[str, Any], raw))

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
        """Async mirror of `ParseJobsResource.list`. See there for full documentation."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = await self._get(
            self._v2_url("/v2/parse/jobs"),
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
        jobs = [normalize_parse_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
        return JobList.build(
            jobs,
            has_more=env.get("has_more"),
            org_id=env.get("org_id"),
            page=env.get("page"),
            page_size=env.get("page_size"),
        )

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: Optional[float] = None,
        raise_on_failure: bool = False,
        _monotonic: Optional[Callable[[], float]] = None,
    ) -> Job:
        """Async mirror of `ParseJobsResource.wait`; sleeps via `anyio.sleep` instead of blocking."""
        return await apoll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )
