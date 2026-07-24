# tests/test_v2_types.py
from __future__ import annotations

from datetime import datetime

from landingai_ade.types.v2 import (
    Job,
    JobError,
    JobStatus,
    V2GroundResult,
    V2ExtractResult,
    V2ParseResponse,
    V2FileUploadResponse,
    V2BuildSchemaResponse,
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


def test_extract_result_new_metadata_and_billing_fields() -> None:
    # model_version / range_units / openapi_spec on metadata, the two new billing
    # counters, and the top-level output_ref all deserialize -- without the
    # legacy `version` field, which current gateway responses no longer send.
    r = V2ExtractResult(
        extraction={},
        extraction_metadata={},
        markdown="d",
        metadata={  # type: ignore[arg-type]
            "job_id": "e1",
            "model_version": "dpt-3-20260710",
            "duration_ms": 5,
            "range_units": "unicode_codepoints",
            "openapi_spec": "https://api.example/openapi.json",
            "billing": {"input_markdown_chars": 100, "output_extraction_chars": 20},
        },
        output_ref="ref-123",
    )
    assert r.metadata.version is None
    assert r.metadata.model_version == "dpt-3-20260710"
    assert r.metadata.range_units == "unicode_codepoints"
    assert r.metadata.openapi_spec is not None and r.metadata.openapi_spec.endswith("openapi.json")
    assert r.metadata.billing is not None
    assert r.metadata.billing.input_markdown_chars == 100
    assert r.metadata.billing.output_extraction_chars == 20
    assert r.output_ref == "ref-123"


def test_parse_response_inline_grounding_and_metadata() -> None:
    # Newer parse responses carry per-node spatial `grounding` ({page, range, box})
    # inline on `structure`, plus `atomic_grounding` on leaves and the renamed
    # `output_markdown_chars` / `range_units` / `openapi_spec` metadata fields.
    # Built via the validating constructor: the legacy page/element `page`/`span`
    # fields are optional, so the current shape (which omits them) validates.
    r = V2ParseResponse(
        markdown="# hi",
        structure={  # type: ignore[arg-type]
            "type": "document",
            "markdown": "# hi",
            "children": [
                {
                    "type": "page",
                    "markdown": "# hi",
                    "grounding": {
                        "page": 1,
                        "range": {"start": 0, "end": 4},
                        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                    },
                    "children": [
                        {
                            "type": "text",
                            "id": "text-0",
                            "markdown": "# hi",
                            "grounding": {
                                "page": 1,
                                "range": {"start": 0, "end": 4},
                                "box": {"xmin": 0.1, "ymin": 0.1, "xmax": 0.9, "ymax": 0.2},
                            },
                            "atomic_grounding": [
                                {
                                    "page": 1,
                                    "range": {"start": 0, "end": 4},
                                    "box": {"xmin": 0.1, "ymin": 0.1, "xmax": 0.9, "ymax": 0.2},
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        metadata={  # type: ignore[arg-type]
            "job_id": "parse-1",
            "model_version": "dpt-3",
            "page_count": 1,
            "failed_pages": [],
            "output_markdown_chars": 4,
            "range_units": "unicode_codepoints",
            "openapi_spec": "https://api.example/openapi.json",
        },
    )
    assert r.structure is not None and r.structure.markdown == "# hi"
    page = r.structure.children[0]
    assert page.grounding is not None and page.grounding.page == 1
    assert page.grounding.range is not None and page.grounding.range.end == 4
    assert page.grounding.box is not None and page.grounding.box.xmax == 1
    assert page.markdown == "# hi"
    # The legacy page/element locators are absent from the current shape.
    assert page.page is None and page.span is None
    el = page.children[0]
    assert el.id == "text-0" and el.markdown == "# hi"
    assert el.span is None
    assert el.grounding is not None and el.grounding.box is not None and el.grounding.box.xmin == 0.1
    assert el.atomic_grounding is not None and len(el.atomic_grounding) == 1
    seg = el.atomic_grounding[0]
    assert seg.range is not None and seg.range.start == 0
    assert r.metadata is not None
    assert r.metadata.output_markdown_chars == 4
    assert r.metadata.range_units == "unicode_codepoints"
    assert r.metadata.openapi_spec is not None


def test_extract_result_metadata_char_counts_warnings_and_schema_violation() -> None:
    # The char counters moved onto `metadata` (from `billing`) upstream, and
    # `schema_violation_error` / `warnings` were added to the result.
    r = V2ExtractResult(
        extraction={},
        extraction_metadata={},
        markdown="d",
        metadata={  # type: ignore[arg-type]
            "job_id": "e1",
            "model_version": "dpt-3",
            "duration_ms": 5,
            "input_markdown_chars": 100,
            "output_extraction_chars": 20,
        },
        schema_violation_error="field 'foo' skipped",
        warnings=[{"code": "partial", "message": "heads up"}],
    )
    assert r.metadata.input_markdown_chars == 100
    assert r.metadata.output_extraction_chars == 20
    assert r.schema_violation_error == "field 'foo' skipped"
    assert r.warnings is not None and r.warnings[0]["code"] == "partial"


def test_ground_result_builds_from_dicts() -> None:
    r = V2GroundResult(
        grounding={
            "invoice_number": [
                {
                    "block_id": "text-1",
                    "type": "text",
                    "grounding": {"page": 1, "range": {"start": 13, "end": 31}},
                }
            ]
        },
        metadata={  # type: ignore[arg-type]
            "job_id": "ground-1",
            "duration_ms": 5,
            "openapi_spec": "https://api.example/openapi.json",
            "billing": {"service_tier": "priority", "total_credits": 2.5},
        },
    )
    assert r.metadata.job_id == "ground-1"
    assert r.metadata.duration_ms == 5
    assert r.metadata.billing is not None and r.metadata.billing.total_credits == 2.5
    assert "invoice_number" in r.grounding


def test_ground_result_retains_unknown_fields() -> None:
    r = V2GroundResult(
        grounding={},
        metadata={"job_id": "g", "duration_ms": 1},  # type: ignore[arg-type]
        surprise=1,  # type: ignore[call-arg]
    )
    assert r.to_dict()["surprise"] == 1


def test_build_schema_response_builds_from_dicts() -> None:
    r = V2BuildSchemaResponse(
        extraction_schema='{"type": "object", "properties": {"revenue": {"type": "string"}}}',
        metadata={  # type: ignore[arg-type]
            "job_id": "extract-bs1",
            "duration_ms": 7,
            "openapi_spec": "https://api.example/openapi.json",
            "warnings": [{"code": "nonconformant_schema", "msg": "heads up"}],
            "billing": {"service_tier": "priority", "total_credits": 1.5},
        },
    )
    assert isinstance(r.extraction_schema, str)
    assert r.metadata.job_id == "extract-bs1"
    assert r.metadata.warnings is not None and r.metadata.warnings[0].code == "nonconformant_schema"
    assert r.metadata.billing is not None and r.metadata.billing.total_credits == 1.5


def test_build_schema_response_retains_unknown_fields() -> None:
    r = V2BuildSchemaResponse(
        extraction_schema="{}",
        metadata={"job_id": "bs", "duration_ms": 1},  # type: ignore[arg-type]
        surprise=1,  # type: ignore[call-arg]
    )
    assert r.to_dict()["surprise"] == 1


def test_file_upload_response() -> None:
    assert V2FileUploadResponse(file_ref="abc").file_ref == "abc"
