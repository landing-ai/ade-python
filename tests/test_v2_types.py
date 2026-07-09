# tests/test_v2_types.py
from __future__ import annotations

from datetime import datetime

from landingai_ade.types.v2 import (
    Job,
    JobError,
    JobStatus,
    V2ExtractResult,
    V2ParseResponse,
    V2FileUploadResponse,
)


def test_job_status_enum_values() -> None:
    assert JobStatus.COMPLETED.value == "completed"
    assert set(JobStatus) >= {
        JobStatus.PENDING,
        JobStatus.PROCESSING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }


def test_job_is_terminal() -> None:
    running = Job(job_id="j1", status=JobStatus.PROCESSING)
    done = Job(job_id="j1", status=JobStatus.COMPLETED)
    failed = Job(job_id="j1", status=JobStatus.FAILED)
    assert running.is_terminal is False
    assert done.is_terminal is True
    assert failed.is_terminal is True


def test_job_holds_typed_result_and_error() -> None:
    job = Job(
        job_id="j1",
        status=JobStatus.FAILED,
        created_at=datetime(2026, 1, 1),
        error=JobError(code="internal_error", message="boom"),
        raw={"org_id": "o1"},
    )
    assert job.error is not None and job.error.code == "internal_error"
    assert job.raw["org_id"] == "o1"


def test_job_raw_default_is_independent_per_instance() -> None:
    job_a = Job(job_id="j1", status=JobStatus.PENDING)
    job_b = Job(job_id="j2", status=JobStatus.PENDING)
    job_a.raw["org_id"] = "o1"
    assert job_b.raw == {}
    assert job_a.raw is not job_b.raw


def test_job_raw_default_is_independent_per_instance_via_construct() -> None:
    # BaseModel.construct()/model_construct() fills in missing fields via
    # field_get_default() without deep-copying. Prove that the raw dict's
    # default_factory=dict still produces a fresh, independent dict per
    # instance on this fast/unvalidated construction path too.
    job_a = Job.construct(job_id="j1", status=JobStatus.PENDING)
    job_b = Job.construct(job_id="j2", status=JobStatus.PENDING)
    job_a.raw["org_id"] = "o1"
    assert job_b.raw == {}
    assert job_a.raw is not job_b.raw


def test_extract_result_parses_nested_metadata() -> None:
    r = V2ExtractResult(
        extraction={"revenue": "1M"},
        extraction_metadata={"revenue": {"value": "1M", "spans": []}},
        markdown="# doc",
        metadata={"job_id": "j1", "version": "extract-1", "duration_ms": 12},  # type: ignore[arg-type]
    )
    assert r.metadata.job_id == "j1"
    assert r.metadata.credit_usage == 0.0  # default


def test_parse_response_builds_from_dicts() -> None:
    # `structure`/`grounding` are typed trees; `metadata` (with nested `billing`)
    # and the whole response build cleanly from plain dicts.
    r = V2ParseResponse(
        markdown="# hi",
        structure={"type": "document", "children": [{"type": "page", "page": 0, "span": [0, 4]}]},  # type: ignore[arg-type]
        metadata={  # type: ignore[arg-type]
            "req_id": "r1",
            "job_id": "j1",
            "model_version": "dpt-3",
            "page_count": 2,
            "failed_pages": [],
            "billing": {"service_tier": "standard", "total_credits": 3.0},
        },
    )
    assert r.markdown == "# hi"
    assert r.structure is not None and r.structure.children[0].page == 0
    assert r.metadata is not None and r.metadata.billing is not None
    assert r.metadata.billing.total_credits == 3.0


def test_parse_response_retains_unknown_fields() -> None:
    r = V2ParseResponse(markdown="# hi", surprise_field="x")  # type: ignore[call-arg]
    assert r.to_dict()["surprise_field"] == "x"

    e = V2ExtractResult(
        extraction={},
        extraction_metadata={},
        markdown="doc",
        metadata={"job_id": "j1", "version": "extract-1", "duration_ms": 12},  # type: ignore[arg-type]
        another_surprise=42,  # type: ignore[call-arg]
    )
    assert e.to_dict()["another_surprise"] == 42


def test_file_upload_response() -> None:
    assert V2FileUploadResponse(file_ref="abc").file_ref == "abc"
