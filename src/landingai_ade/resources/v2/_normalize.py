# src/landingai_ade/resources/v2/_normalize.py
from __future__ import annotations

from typing import Any, Dict, Union, Mapping, Optional, cast
from datetime import datetime

from ..._types import StrBytesIntFloat
from ..._utils import parse_datetime
from ...types.v2 import Job, JobError, JobStatus, V2ExtractResult, V2ParseResponse

__all__ = ["normalize_parse_job", "normalize_extract_job"]


def _ts(value: Optional[Union[datetime, StrBytesIntFloat]]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return parse_datetime(value)  # handles both epoch ints and ISO strings
    except (ValueError, TypeError):
        return None


def _progress(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _status(raw: Mapping[str, Any]) -> JobStatus:
    value = raw.get("status")
    if value is None:
        return JobStatus.PENDING
    try:
        return JobStatus(value)
    except ValueError:
        # Unknown/renamed status from the gateway: don't crash the whole
        # normalizer. The original raw status is still available via job.raw.
        return JobStatus.PENDING


def normalize_parse_job(raw: Mapping[str, Any]) -> Job:
    status = _status(raw)
    data = raw.get("data")
    result = V2ParseResponse(**cast(Dict[str, Any], data)) if isinstance(data, Mapping) else None

    error = None
    reason = raw.get("failure_reason")
    if reason:
        error = JobError(message=str(reason))

    created = raw.get("created_at")
    created = created if created is not None else raw.get("received_at")

    return Job(
        job_id=str(raw["job_id"]),
        status=status,
        created_at=_ts(created),
        completed_at=None,  # parse envelope has no completed_at
        progress=_progress(raw.get("progress")),
        result=result,
        error=error,
        raw=dict(raw),
    )


def normalize_extract_job(raw: Mapping[str, Any]) -> Job:
    status = _status(raw)
    payload = raw.get("result")
    result = V2ExtractResult(**cast(Dict[str, Any], payload)) if isinstance(payload, Mapping) else None

    error = None
    err = raw.get("error")
    if isinstance(err, Mapping):
        err = cast(Dict[str, Any], err)
        error = JobError(code=err.get("code"), message=err.get("message"))
    elif raw.get("failure_reason"):  # extract *list* uses failure_reason
        error = JobError(message=str(raw["failure_reason"]))

    return Job(
        job_id=str(raw["job_id"]),
        status=status,
        created_at=_ts(raw.get("created_at")),
        completed_at=_ts(raw.get("completed_at")),
        progress=_progress(raw.get("progress")),
        result=result,
        error=error,
        raw=dict(raw),
    )
