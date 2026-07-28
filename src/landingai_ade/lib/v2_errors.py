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
    """A synchronous /v2 call exceeded the server wait window (504).

    The server cancels the work; use the endpoint's matching async jobs route
    (`client.v2.parse_jobs` / `extract_jobs` / `build_schema_jobs`), then
    `.wait(...)`, for long-running inputs. The raised message names the route
    for the specific endpoint that timed out."""


class JobWaitTimeoutError(LandingAiadeError):
    """`wait()` gave up before the job reached a terminal state."""


class JobFailedError(LandingAiadeError):
    """A job reached a terminal `failed`/`cancelled` state (raise_on_failure=True)."""


def raise_if_sync_timeout(exc: APIStatusError, *, jobs_resource: str = "") -> None:
    """Re-raise a 504 as `V2SyncTimeoutError` pointing at this endpoint's async route.

    `jobs_resource` is the `client.v2.<name>` async jobs resource to recommend for
    the timed-out endpoint (e.g. ``"parse_jobs"``, ``"extract_jobs"``,
    ``"build_schema_jobs"``). Optional and keyword-only so adding
    it stays backward compatible; when omitted the message points generically at the
    endpoint's async jobs route rather than naming a possibly-wrong resource. Always
    pass it for a new endpoint so the remediation is specific.
    """
    if exc.response.status_code == 504:
        route = f"`client.v2.{jobs_resource}.create(...)`" if jobs_resource else "the endpoint's async jobs route"
        raise V2SyncTimeoutError(
            "The synchronous request timed out (HTTP 504). The server cancels the work on "
            f"timeout — use {route}, then `.wait(...)`, for long-running inputs."
        ) from exc
