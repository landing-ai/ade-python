# src/landingai_ade/resources/v2/_base.py
from __future__ import annotations

from typing import Any, List, Callable, Optional, Awaitable

import anyio

from ...types.v2 import Job
from ...lib.v2_errors import JobFailedError, JobWaitTimeoutError

__all__ = [
    "V2ResourceMixin",
    "JobList",
    "poll_until_terminal",
    "apoll_until_terminal",
    "DEFAULT_POLL_INITIAL",
    "DEFAULT_POLL_MAX",
    "DEFAULT_POLL_FACTOR",
    "DEFAULT_WAIT_TIMEOUT",
]

DEFAULT_POLL_INITIAL = 1.0
DEFAULT_POLL_MAX = 10.0
DEFAULT_POLL_FACTOR = 1.5
DEFAULT_WAIT_TIMEOUT = 600.0


class V2ResourceMixin:
    """Shared helpers for V2 sub-resources, which target an absolute (non-V1) host.

    ``_client`` is provided by whichever of ``SyncAPIResource``/``AsyncAPIResource``
    this mixes in alongside; it is declared here as ``Any`` only so this mixin's own
    methods type-check in isolation.
    """

    _client: Any  # LandingAIADE | AsyncLandingAIADE

    def _v2_url(self, path: str) -> str:
        return f"{self._client._v2_base_url}{path}"


def _next_delay(current: float, poll_interval: Optional[float]) -> float:
    if poll_interval is not None:
        return poll_interval
    return min(current * DEFAULT_POLL_FACTOR, DEFAULT_POLL_MAX)


def _raise_if_failed(job: Job, *, raise_on_failure: bool) -> None:
    if raise_on_failure and job.error is not None:
        raise JobFailedError(
            f"Job {job.job_id} ended {job.status.value}: {job.error.message or job.error.code or 'unknown error'}"
        )


def poll_until_terminal(
    get_job: Callable[[], Job],
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    timeout: float,
    poll_interval: Optional[float],
    raise_on_failure: bool,
) -> Job:
    """Poll ``get_job`` with backoff until the job reaches a terminal state.

    ``monotonic``/``sleep`` are injected (rather than reading ``time`` directly) so
    tests can drive a fake clock without real time passing; production callers pass
    ``time.monotonic``/``time.sleep``.
    """
    deadline = monotonic() + timeout
    delay = poll_interval if poll_interval is not None else DEFAULT_POLL_INITIAL
    while True:
        job = get_job()
        if job.is_terminal:
            _raise_if_failed(job, raise_on_failure=raise_on_failure)
            return job
        if monotonic() >= deadline:
            raise JobWaitTimeoutError(
                f"Job {job.job_id} did not finish within {timeout}s (last status: {job.status.value})."
            )
        sleep(min(delay, max(0.0, deadline - monotonic())))
        delay = _next_delay(delay, poll_interval)


async def apoll_until_terminal(
    get_job: Callable[[], Awaitable[Job]],
    *,
    monotonic: Callable[[], float],
    timeout: float,
    poll_interval: Optional[float],
    raise_on_failure: bool,
) -> Job:
    """Async mirror of :func:`poll_until_terminal`; sleeps via ``anyio.sleep``."""
    deadline = monotonic() + timeout
    delay = poll_interval if poll_interval is not None else DEFAULT_POLL_INITIAL
    while True:
        job = await get_job()
        if job.is_terminal:
            _raise_if_failed(job, raise_on_failure=raise_on_failure)
            return job
        if monotonic() >= deadline:
            raise JobWaitTimeoutError(
                f"Job {job.job_id} did not finish within {timeout}s (last status: {job.status.value})."
            )
        await anyio.sleep(min(delay, max(0.0, deadline - monotonic())))
        delay = _next_delay(delay, poll_interval)


def _cast_opt_str(value: object) -> Optional[str]:
    return value if isinstance(value, str) else None


class JobList(List[Job]):
    """A list of normalized jobs plus the pagination envelope (``has_more``, ``org_id``, ...)."""

    has_more: bool = False
    org_id: Optional[str] = None
    page: Optional[int] = None
    page_size: Optional[int] = None

    @classmethod
    def build(cls, jobs: List[Job], **envelope: object) -> "JobList":
        out = cls(jobs)
        hm = envelope.get("has_more")
        out.has_more = hm if isinstance(hm, bool) else False
        out.org_id = _cast_opt_str(envelope.get("org_id"))
        page = envelope.get("page")
        page_size = envelope.get("page_size")
        out.page = page if isinstance(page, int) else None
        out.page_size = page_size if isinstance(page_size, int) else None
        return out
