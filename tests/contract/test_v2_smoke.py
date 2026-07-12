from __future__ import annotations

import os
from typing import Iterator
from pathlib import Path

import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import JobStatus, V2ExtractResult, V2ParseResponse

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
    assert res.metadata.version


def test_extract_jobs(staging_client: LandingAIADE) -> None:
    job = staging_client.v2.extract_jobs.create(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
    done = staging_client.v2.extract_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2ExtractResult)


def test_parse_sync(staging_client: LandingAIADE) -> None:
    pdf = Path(__file__).parent / "sample.pdf"
    resp = staging_client.v2.parse(document=pdf)
    assert isinstance(resp, V2ParseResponse)
    assert isinstance(resp.markdown, str)
    assert resp.markdown


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
