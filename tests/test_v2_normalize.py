# tests/test_v2_normalize.py
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime, timezone

from landingai_ade.types.v2 import JobStatus, V2ExtractResult, V2ParseResponse
from landingai_ade.resources.v2._normalize import normalize_parse_job, normalize_extract_job


def test_normalize_parse_job_epoch_and_data() -> None:
    raw = {
        "job_id": "p1",
        "status": "completed",
        "received_at": 1_700_000_000,
        "created_at": 1_700_000_005,
        "progress": 1.0,
        "org_id": "o1",
        "output_url": None,
        "data": {"markdown": "# hi", "metadata": {"job_id": "p1", "page_count": 1}},
    }
    job = normalize_parse_job(raw)
    assert job.job_id == "p1"
    assert job.status is JobStatus.COMPLETED
    assert job.created_at is not None and job.created_at.year == 2023
    assert isinstance(job.result, V2ParseResponse)
    assert job.result.markdown == "# hi"
    assert job.error is None
    assert job.raw["org_id"] == "o1"  # envelope-only fields preserved


def test_normalize_parse_job_preserves_epoch_zero_created_at() -> None:
    # created_at == 0 (epoch) is falsy but must NOT be treated as missing;
    # it must not fall back to received_at. received_at=123 is also within
    # 1970, so we assert the exact instant (not just the year) to ensure the
    # received_at fallback (00:02:03) wasn't used instead of epoch (00:00:00).
    raw = {"job_id": "p", "status": "pending", "created_at": 0, "received_at": 123}
    job = normalize_parse_job(raw)
    assert job.created_at is not None
    assert job.created_at.year == 1970
    assert job.created_at == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_normalize_parse_job_failure_reason() -> None:
    raw = {"job_id": "p2", "status": "failed", "failure_reason": "bad pdf", "created_at": 1_700_000_000}
    job = normalize_parse_job(raw)
    assert job.status is JobStatus.FAILED
    assert job.error is not None and job.error.message == "bad pdf"
    assert job.result is None


def test_normalize_extract_job_iso_and_result() -> None:
    raw: Dict[str, Any] = {
        "job_id": "e1",
        "status": "completed",
        "created_at": "2026-01-02T03:04:05Z",
        "completed_at": "2026-01-02T03:04:09Z",
        "result": {
            "extraction": {"revenue": "1M"},
            "extraction_metadata": {"revenue": {"value": "1M", "spans": []}},
            "markdown": "# doc",
            "metadata": {"job_id": "e1", "version": "extract-1", "duration_ms": 10},
        },
    }
    job = normalize_extract_job(raw)
    assert job.status is JobStatus.COMPLETED
    assert job.created_at is not None and job.created_at.year == 2026
    assert job.completed_at is not None
    assert isinstance(job.result, V2ExtractResult)
    assert job.result.metadata.version == "extract-1"


def test_normalize_extract_job_error_object() -> None:
    raw = {"job_id": "e2", "status": "failed", "error": {"code": "internal_error", "message": "boom"}}
    job = normalize_extract_job(raw)
    assert job.status is JobStatus.FAILED
    assert job.error is not None and job.error.code == "internal_error"


def test_normalize_parse_job_minimal_create_envelope_defaults_to_pending() -> None:
    # Live /v2/parse/jobs create (202) response is minimal: only job_id, no status.
    raw = {"job_id": "parse-api-x"}
    job = normalize_parse_job(raw)
    assert job.job_id == "parse-api-x"
    assert job.status is JobStatus.PENDING
    assert job.result is None
    assert job.error is None


def test_normalize_extract_job_minimal_create_envelope_defaults_to_pending() -> None:
    raw = {"job_id": "v2-extract-x"}
    job = normalize_extract_job(raw)
    assert job.job_id == "v2-extract-x"
    assert job.status is JobStatus.PENDING
    assert job.result is None
    assert job.error is None
