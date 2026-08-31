from __future__ import annotations

import json
import time
from typing import Any, Mapping, Callable, Iterable, Optional, cast
from typing_extensions import Literal

import httpx

from ._base import DEFAULT_WAIT_TIMEOUT, JobList, V2ResourceMixin, poll_until_terminal, apoll_until_terminal
from ..._files import deepcopy_with_paths
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._utils import is_given, extract_files
from ...types.v2 import Job, V2ClassifyResponse
from ._normalize import normalize_classify_job
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import APIStatusError
from ..._base_client import make_request_options
from ...lib.v2_errors import raise_if_sync_timeout

__all__ = ["ClassifyResource", "AsyncClassifyResource", "ClassifyJobsResource", "AsyncClassifyJobsResource"]


def _build_classify_body(
    document: object,
    document_url: object,
    classes: Iterable[Mapping[str, object]],
    model: object,
) -> dict[str, Any]:
    # `classes` is a JSON-encoded string form field per the contract: each entry
    # is an object with a `class` name and an optional `description`.
    body: dict[str, Any] = {"classes": json.dumps([dict(entry) for entry in classes])}
    # Multipart requests aren't run through `maybe_transform`, so drop unset
    # `omit`/`not_given` sentinels (and explicit `None`) here so they aren't
    # serialized as form fields.
    for key, value in (
        ("document", document),
        ("document_url", document_url),
        ("model", model),
    ):
        if is_given(value) and value is not None:
            body[key] = value
    return body


class ClassifyResource(V2ResourceMixin, SyncAPIResource):
    def run(
        self,
        *,
        classes: Iterable[Mapping[str, object]],
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ClassifyResponse:
        """Classify each page of a document into classes you define, synchronously
        against the V2 (ADE) `/v1/classify` endpoint (multipart body).

        Returns one predicted class per page. Raises `V2SyncTimeoutError` when the
        server times out the synchronous request (HTTP 504); use the async jobs
        route (`client.v2.classify_jobs`) for long-running documents in that case.

        Args:
          classes: The candidate classes for each page. Each entry is a mapping with a
              `class` name and an optional `description`. Only one class is assigned per
              page; unclassifiable pages receive `"unknown"`. Sent to the server as a
              JSON-encoded string form field.

          document: A file to be classified. Either this parameter or `document_url` must be
              provided.

          document_url: The URL to the file to be classified. Either this parameter or `document`
              must be provided.

          model: The classify pipeline version to use (e.g. `classify-latest`).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_with_paths(
            _build_classify_body(document, document_url, classes, model),
            [["document"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        # It should be noted that the actual Content-Type header that will be
        # sent to the server will contain a `boundary` parameter, e.g.
        # multipart/form-data; boundary=---abc--
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        try:
            return self._post(
                self._v2_url("/v1/classify"),
                body=body,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ClassifyResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc, jobs_resource="classify_jobs")
            raise


class AsyncClassifyResource(V2ResourceMixin, AsyncAPIResource):
    async def run(
        self,
        *,
        classes: Iterable[Mapping[str, object]],
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ClassifyResponse:
        """Async mirror of `ClassifyResource.run`. See there for full documentation."""
        body = deepcopy_with_paths(
            _build_classify_body(document, document_url, classes, model),
            [["document"]],
        )
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        try:
            return await self._post(
                self._v2_url("/v1/classify"),
                body=body,
                files=files,
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=V2ClassifyResponse,
            )
        except APIStatusError as exc:
            raise_if_sync_timeout(exc, jobs_resource="classify_jobs")
            raise


class ClassifyJobsResource(V2ResourceMixin, SyncAPIResource):
    def create(
        self,
        *,
        classes: Iterable[Mapping[str, object]],
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Create an asynchronous classify job against `/v1/classify/jobs`.

        Returns a normalized `Job` immediately (typically `pending`). Poll for
        completion via `.get(job_id)`, or block until the job is terminal with
        `.wait(job_id)`.

        Args:
          classes: The candidate classes for each page. Each entry is a mapping with a
              `class` name and an optional `description`. Sent to the server as a
              JSON-encoded string form field.

          document: A file to be classified. Either this parameter or `document_url` must be
              provided.

          document_url: The URL to the file to be classified. Either this parameter or `document`
              must be provided.

          model: The classify pipeline version to use (e.g. `classify-latest`).

          service_tier: Service tier for the job: ``standard`` or ``priority``.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = _build_classify_body(document, document_url, classes, model)
        if is_given(service_tier) and service_tier is not None:
            body["service_tier"] = service_tier
        body = deepcopy_with_paths(body, [["document"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        raw = self._post(
            self._v2_url("/v1/classify/jobs"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_classify_job(cast(Mapping[str, Any], raw))

    def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Get the current status of an async classify job by `job_id`."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = self._get(
            self._v2_url(f"/v1/classify/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_classify_job(cast(Mapping[str, Any], raw))

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
        """List async classify jobs associated with your API key, newest first."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = self._get(
            self._v2_url("/v1/classify/jobs"),
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
        jobs = [normalize_classify_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
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
        set and the job ends failed with an error attached.

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


class AsyncClassifyJobsResource(V2ResourceMixin, AsyncAPIResource):
    async def create(
        self,
        *,
        classes: Iterable[Mapping[str, object]],
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        service_tier: Optional[Literal["standard", "priority"]] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `ClassifyJobsResource.create`. See there for full documentation."""
        body = _build_classify_body(document, document_url, classes, model)
        if is_given(service_tier) and service_tier is not None:
            body["service_tier"] = service_tier
        body = deepcopy_with_paths(body, [["document"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["document"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        raw = await self._post(
            self._v2_url("/v1/classify/jobs"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_classify_job(cast(Mapping[str, Any], raw))

    async def get(
        self,
        job_id: str,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> Job:
        """Async mirror of `ClassifyJobsResource.get`. See there for full documentation."""
        if not job_id:
            raise ValueError(f"Expected a non-empty value for `job_id` but received {job_id!r}")
        raw = await self._get(
            self._v2_url(f"/v1/classify/jobs/{job_id}"),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=cast("type[Any]", object),
        )
        return normalize_classify_job(cast(Mapping[str, Any], raw))

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
        """Async mirror of `ClassifyJobsResource.list`. See there for full documentation."""
        query = {
            key: value
            for key, value in {"page": page, "page_size": page_size, "status": status}.items()
            if is_given(value) and value is not None
        }
        raw = await self._get(
            self._v2_url("/v1/classify/jobs"),
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
        jobs = [normalize_classify_job(cast(Mapping[str, Any], item)) for item in env.get("jobs", [])]
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
        """Async mirror of `ClassifyJobsResource.wait`; sleeps via `anyio.sleep` instead of blocking."""
        return await apoll_until_terminal(
            lambda: self.get(job_id),
            monotonic=_monotonic or time.monotonic,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_on_failure=raise_on_failure,
        )
