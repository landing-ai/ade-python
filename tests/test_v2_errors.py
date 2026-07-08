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
    request = httpx.Request("POST", "https://aide.landing.ai/v2/extract")
    response = httpx.Response(code, request=request)
    return APIStatusError("err", response=response, body=None)


def test_raise_if_sync_timeout_converts_504() -> None:
    with pytest.raises(V2SyncTimeoutError):
        raise_if_sync_timeout(_status_error(504))


def test_raise_if_sync_timeout_ignores_other_codes() -> None:
    raise_if_sync_timeout(_status_error(500))  # returns without raising


def test_error_hierarchy() -> None:
    from landingai_ade._exceptions import LandingAiadeError

    assert issubclass(V2SyncTimeoutError, LandingAiadeError)
    assert issubclass(JobWaitTimeoutError, LandingAiadeError)
    assert issubclass(JobFailedError, LandingAiadeError)
