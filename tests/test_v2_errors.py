# tests/test_v2_errors.py
from __future__ import annotations

import httpx
import pytest

from landingai_ade._exceptions import APIStatusError
from landingai_ade.lib.v2_errors import (
    JobFailedError,
    V2SyncTimeoutError,
    JobWaitTimeoutError,
    raise_if_sync_timeout,
)


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.ade.landing.ai/v2/extract")
    response = httpx.Response(code, request=request)
    return APIStatusError("err", response=response, body=None)


def test_raise_if_sync_timeout_converts_504() -> None:
    with pytest.raises(V2SyncTimeoutError) as exc_info:
        raise_if_sync_timeout(_status_error(504), jobs_resource="build_schema_jobs")
    # The remediation names the endpoint's own async route, not a hardcoded parse/extract one.
    assert "build_schema_jobs" in str(exc_info.value)


def test_raise_if_sync_timeout_ignores_other_codes() -> None:
    raise_if_sync_timeout(_status_error(500), jobs_resource="parse_jobs")  # returns without raising


def test_error_hierarchy() -> None:
    from landingai_ade._exceptions import LandingAiadeError

    assert issubclass(V2SyncTimeoutError, LandingAiadeError)
    assert issubclass(JobWaitTimeoutError, LandingAiadeError)
    assert issubclass(JobFailedError, LandingAiadeError)
