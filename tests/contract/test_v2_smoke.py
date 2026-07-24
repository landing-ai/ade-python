from __future__ import annotations

import os
import json
from typing import Iterator
from pathlib import Path

import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import (
    JobStatus,
    V2GroundResult,
    V2ExtractResult,
    V2ParseResponse,
    V2BuildSchemaResponse,
)

pytestmark = pytest.mark.contract

STAGING_KEY = os.environ.get("LANDINGAI_ADE_STAGING_APIKEY")

# A tiny self-contained markdown document so extract/files can run without any file.
SAMPLE_MARKDOWN = "# Acme Inc. — Q1 Report\n\nTotal revenue for the quarter was **$1,250,000**.\n"


class RevenueSchema(BaseModel):
    """Demonstrates passing a pydantic model as the extract schema."""

    revenue: str = Field(description="The total revenue figure, verbatim")
    company: str = Field(description="The company name")


@pytest.fixture()
def staging_client() -> Iterator[LandingAIADE]:
    if not STAGING_KEY:
        pytest.skip("LANDINGAI_ADE_STAGING_APIKEY not set")
    # Context-managed so the underlying HTTP client is closed in teardown (no socket leak).
    with LandingAIADE(apikey=STAGING_KEY, environment="staging") as client:
        yield client


def test_files_upload(staging_client: LandingAIADE) -> None:
    file_ref = staging_client.v2.files.upload(file=("doc.md", SAMPLE_MARKDOWN.encode(), "text/markdown"))
    assert isinstance(file_ref, str)
    assert file_ref


def test_extract_sync(staging_client: LandingAIADE) -> None:
    res = staging_client.v2.extract(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
    assert isinstance(res, V2ExtractResult)
    assert isinstance(res.extraction, dict)
    assert res.extraction
    # `version` was renamed to `model_version` upstream; the current gateway
    # populates `model_version`.
    assert res.metadata.model_version


def test_extract_jobs(staging_client: LandingAIADE) -> None:
    job = staging_client.v2.extract_jobs.create(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
    done = staging_client.v2.extract_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2ExtractResult)


def test_build_schema_sync(staging_client: LandingAIADE) -> None:
    # Generate a JSON Schema from a source markdown document and a prompt.
    res = staging_client.v2.build_schema(
        markdowns=[SAMPLE_MARKDOWN],
        prompt="Capture the total revenue figure and the company name.",
    )
    assert isinstance(res, V2BuildSchemaResponse)
    assert isinstance(res.extraction_schema, str) and res.extraction_schema
    # The generated schema is a JSON Schema serialized as a string.
    assert isinstance(json.loads(res.extraction_schema), dict)


def test_build_schema_jobs(staging_client: LandingAIADE) -> None:
    job = staging_client.v2.build_schema_jobs.create(
        markdowns=[SAMPLE_MARKDOWN],
        prompt="Capture the total revenue figure and the company name.",
    )
    done = staging_client.v2.build_schema_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2BuildSchemaResponse)


def test_parse_sync(staging_client: LandingAIADE) -> None:
    pdf = Path(__file__).parent / "sample.pdf"
    resp = staging_client.v2.parse(document=pdf)
    assert isinstance(resp, V2ParseResponse)
    assert isinstance(resp.markdown, str)
    assert resp.markdown


def test_parse_sync_inline_grounding_and_metadata(staging_client: LandingAIADE) -> None:
    # Exercise the current parse surface: `inline_markdown` option, per-node
    # spatial `grounding` ({page, range, box}) inline on `structure`, and the
    # renamed `output_markdown_chars` / `range_units` metadata fields.
    pdf = Path(__file__).parent / "sample.pdf"
    resp = staging_client.v2.parse(document=pdf, options={"inline_markdown": True})
    assert isinstance(resp, V2ParseResponse)
    assert resp.structure is not None and resp.structure.children
    page = resp.structure.children[0]
    assert page.grounding is not None and page.grounding.range is not None
    assert page.grounding.box is not None
    assert resp.metadata is not None
    assert resp.metadata.range_units == "unicode_codepoints"
    assert resp.metadata.output_markdown_chars is not None


def test_ground_sync(staging_client: LandingAIADE) -> None:
    # Ground is a stateless join: parse the doc, extract against it, then ground
    # the extraction back onto the parse structure the markdown came from.
    parsed = staging_client.v2.parse(document=Path(__file__).parent / "sample.pdf")
    assert parsed.structure is not None
    extracted = staging_client.v2.extract(schema=RevenueSchema, markdown=parsed.markdown or "")
    grounded = staging_client.v2.ground(
        extraction_metadata=extracted.extraction_metadata,
        structure=parsed.structure,
    )
    assert isinstance(grounded, V2GroundResult)
    assert isinstance(grounded.grounding, dict)
    assert grounded.metadata.job_id


def test_ground_jobs(staging_client: LandingAIADE) -> None:
    parsed = staging_client.v2.parse(document=Path(__file__).parent / "sample.pdf")
    assert parsed.structure is not None
    extracted = staging_client.v2.extract(schema=RevenueSchema, markdown=parsed.markdown or "")
    job = staging_client.v2.ground_jobs.create(
        extraction_metadata=extracted.extraction_metadata,
        structure=parsed.structure,
    )
    done = staging_client.v2.ground_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2GroundResult)


def test_parse_jobs(staging_client: LandingAIADE) -> None:
    pdf = Path(__file__).parent / "sample.pdf"
    job = staging_client.v2.parse_jobs.create(document=pdf)
    done = staging_client.v2.parse_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    # Assert the normalized job result, not just the terminal status, so this test
    # actually covers the parse-job response contract (data -> V2ParseResponse).
    assert isinstance(done.result, V2ParseResponse)
    assert isinstance(done.result.markdown, str)
    assert done.result.markdown
