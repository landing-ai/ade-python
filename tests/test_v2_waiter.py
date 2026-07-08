from __future__ import annotations

from typing import List, Callable, Optional

import pytest

from landingai_ade.types.v2 import Job, JobError, JobStatus
from landingai_ade.resources.v2 import _base
from landingai_ade.lib.v2_errors import JobFailedError, JobWaitTimeoutError
from landingai_ade.resources.v2._base import (
    JobList,
    poll_until_terminal,
    apoll_until_terminal,
)


def _job(status: JobStatus, job_id: str = "job-1", error: Optional[JobError] = None) -> Job:
    return Job(job_id=job_id, status=status, error=error)


class FakeClock:
    """A fake monotonic clock + sleep recorder so waiter tests never sleep for real."""

    def __init__(self, times: Optional[List[float]] = None, start: float = 0.0) -> None:
        self._times = list(times) if times is not None else None
        self.t = start
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        if self._times is not None:
            if len(self._times) > 1:
                return self._times.pop(0)
            return self._times[0]
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)


def _get_job_sequence(jobs: List[Job]) -> Callable[[], Job]:
    it = iter(jobs)

    def _get() -> Job:
        return next(it)

    return _get


# --------------------------------------------------------------------------
# poll_until_terminal (sync)
# --------------------------------------------------------------------------


def test_poll_until_terminal_returns_immediately_when_already_terminal() -> None:
    clock = FakeClock()
    job = _job(JobStatus.COMPLETED)

    result = poll_until_terminal(
        _get_job_sequence([job]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        timeout=10.0,
        poll_interval=None,
        raise_on_failure=False,
    )

    assert result is job
    assert clock.sleeps == []


def test_poll_until_terminal_backs_off_between_polls() -> None:
    clock = FakeClock()
    jobs = [_job(JobStatus.PENDING), _job(JobStatus.PROCESSING), _job(JobStatus.COMPLETED)]

    result = poll_until_terminal(
        _get_job_sequence(jobs),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        timeout=100.0,
        poll_interval=None,
        raise_on_failure=False,
    )

    assert result.status == JobStatus.COMPLETED
    # DEFAULT_POLL_INITIAL=1.0, next delay = min(1.0*1.5, 10.0) = 1.5
    assert clock.sleeps == [1.0, 1.5]


def test_poll_until_terminal_uses_fixed_poll_interval() -> None:
    clock = FakeClock()
    jobs = [_job(JobStatus.PENDING), _job(JobStatus.PENDING), _job(JobStatus.COMPLETED)]

    poll_until_terminal(
        _get_job_sequence(jobs),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        timeout=100.0,
        poll_interval=2.0,
        raise_on_failure=False,
    )

    assert clock.sleeps == [2.0, 2.0]


def test_poll_until_terminal_raises_timeout() -> None:
    # First monotonic() call establishes the deadline (0 + timeout); the second call
    # (post get_job) reports we're already past it, so we raise without sleeping.
    clock = FakeClock(times=[0.0, 100.0])
    job = _job(JobStatus.PENDING)

    with pytest.raises(JobWaitTimeoutError, match="did not finish within"):
        poll_until_terminal(
            _get_job_sequence([job]),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            timeout=10.0,
            poll_interval=None,
            raise_on_failure=False,
        )


def test_poll_until_terminal_raise_on_failure_true_raises() -> None:
    clock = FakeClock()
    job = _job(JobStatus.FAILED, error=JobError(code="bad_input", message="boom"))

    with pytest.raises(JobFailedError, match="boom"):
        poll_until_terminal(
            _get_job_sequence([job]),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            timeout=10.0,
            poll_interval=None,
            raise_on_failure=True,
        )


def test_poll_until_terminal_raise_on_failure_false_returns_job() -> None:
    clock = FakeClock()
    job = _job(JobStatus.FAILED, error=JobError(code="bad_input", message="boom"))

    result = poll_until_terminal(
        _get_job_sequence([job]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        timeout=10.0,
        poll_interval=None,
        raise_on_failure=False,
    )

    assert result is job
    assert result.status == JobStatus.FAILED


# --------------------------------------------------------------------------
# apoll_until_terminal (async) -- anyio.sleep is patched so no real time passes.
# --------------------------------------------------------------------------


async def test_apoll_until_terminal_returns_immediately_when_already_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(_base.anyio, "sleep", clock.asleep)
    job = _job(JobStatus.COMPLETED)

    async def _get() -> Job:
        return job

    result = await apoll_until_terminal(
        _get,
        monotonic=clock.monotonic,
        timeout=10.0,
        poll_interval=None,
        raise_on_failure=False,
    )

    assert result is job
    assert clock.sleeps == []


async def test_apoll_until_terminal_backs_off_between_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(_base.anyio, "sleep", clock.asleep)
    jobs = iter([_job(JobStatus.PENDING), _job(JobStatus.PROCESSING), _job(JobStatus.COMPLETED)])

    async def _get() -> Job:
        return next(jobs)

    result = await apoll_until_terminal(
        _get,
        monotonic=clock.monotonic,
        timeout=100.0,
        poll_interval=None,
        raise_on_failure=False,
    )

    assert result.status == JobStatus.COMPLETED
    assert clock.sleeps == [1.0, 1.5]


async def test_apoll_until_terminal_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(times=[0.0, 100.0])
    monkeypatch.setattr(_base.anyio, "sleep", clock.asleep)
    job = _job(JobStatus.PENDING)

    async def _get() -> Job:
        return job

    with pytest.raises(JobWaitTimeoutError, match="did not finish within"):
        await apoll_until_terminal(
            _get,
            monotonic=clock.monotonic,
            timeout=10.0,
            poll_interval=None,
            raise_on_failure=False,
        )


async def test_apoll_until_terminal_raise_on_failure_true_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(_base.anyio, "sleep", clock.asleep)
    job = _job(JobStatus.CANCELLED, error=JobError(message="cancelled by user"))

    async def _get() -> Job:
        return job

    with pytest.raises(JobFailedError, match="cancelled by user"):
        await apoll_until_terminal(
            _get,
            monotonic=clock.monotonic,
            timeout=10.0,
            poll_interval=None,
            raise_on_failure=True,
        )


# --------------------------------------------------------------------------
# JobList
# --------------------------------------------------------------------------


def test_joblist_build_defaults_when_envelope_empty() -> None:
    jobs = [_job(JobStatus.COMPLETED, job_id="a"), _job(JobStatus.PENDING, job_id="b")]

    result = JobList.build(jobs)

    assert isinstance(result, list)
    assert list(result) == jobs
    assert result.has_more is False
    assert result.org_id is None
    assert result.page is None
    assert result.page_size is None


def test_joblist_build_captures_full_envelope() -> None:
    jobs = [_job(JobStatus.COMPLETED)]

    result = JobList.build(jobs, has_more=True, org_id="org-123", page=2, page_size=50)

    assert result.has_more is True
    assert result.org_id == "org-123"
    assert result.page == 2
    assert result.page_size == 50


def test_joblist_build_ignores_wrong_typed_envelope_values() -> None:
    jobs = [_job(JobStatus.COMPLETED)]

    result = JobList.build(jobs, org_id=123, page="two", page_size=None)

    assert result.org_id is None
    assert result.page is None
    assert result.page_size is None


def test_joblist_build_has_more_rejects_truthy_string() -> None:
    # A wrong-typed (but truthy) string must not be coerced via bool(); only a
    # real bool should ever set has_more to True.
    result = JobList.build([], has_more="false")

    assert result.has_more is False


def test_joblist_build_has_more_accepts_real_bool() -> None:
    result = JobList.build([], has_more=True)

    assert result.has_more is True
