# src/landingai_ade/lib/v2_errors.py
from __future__ import annotations

from .._exceptions import APIStatusError, LandingAiadeError

__all__ = [
    "V2SyncTimeoutError",
    "JobWaitTimeoutError",
    "JobFailedError",
    "raise_if_sync_timeout",
]


class V2SyncTimeoutError(LandingAiadeError):
    """A synchronous /v2/parse or /v2/extract call exceeded the server wait window (504).

    The server cancels the work; use the async jobs route (`*_jobs.create` + `wait`)
    for long-running documents."""


class JobWaitTimeoutError(LandingAiadeError):
    """`wait()` gave up before the job reached a terminal state."""


class JobFailedError(LandingAiadeError):
    """A job reached a terminal `failed`/`cancelled` state (raise_on_failure=True)."""


def raise_if_sync_timeout(exc: APIStatusError) -> None:
    if exc.response.status_code == 504:
        raise V2SyncTimeoutError(
            "The synchronous request timed out (HTTP 504). The server cancels the work on "
            "timeout — use the async jobs route (`.jobs.create(...)` then `.jobs.wait(...)`) "
            "for long-running documents."
        ) from exc
