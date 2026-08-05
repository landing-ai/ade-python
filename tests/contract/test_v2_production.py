from __future__ import annotations

import os
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
)

# `production` marker, same as the V1 production suite: excluded from the default `pytest`
# run and the staging contract gate, run only by e2e-production.yml (manual dispatch + the
# release gate). Mirrors tests/contract/test_v2_smoke.py against the LIVE PRODUCTION V2 API
# (api.ade.landing.ai) and spends real credits. The soft-hidden `v2.build_schema` surface is
# intentionally not covered here, matching the staging smoke.
pytestmark = pytest.mark.production

PRODUCTION_KEY = os.environ.get("LANDINGAI_ADE_PRODUCTION_APIKEY")

SAMPLE_PDF = Path(__file__).parent / "sample.pdf"

# A tiny self-contained markdown document so extract can run without a parse first.
SAMPLE_MARKDOWN = "# Acme Inc. — Q1 Report\n\nTotal revenue for the quarter was **$1,250,000**.\n"


class RevenueSchema(BaseModel):
    """Passing a pydantic model as the extract schema (V2 coerces it to JSON Schema)."""

    revenue: str = Field(description="The total revenue figure, verbatim")
    company: str = Field(description="The company name")


@pytest.fixture(scope="module")
def production_client() -> Iterator[LandingAIADE]:
    if not PRODUCTION_KEY:
        pytest.skip("LANDINGAI_ADE_PRODUCTION_APIKEY not set")
    # `environment="production"` routes V2 to api.ade.landing.ai (the V1 api.va host is unused
    # by this suite). The same key authenticates both hosts.
    with LandingAIADE(apikey=PRODUCTION_KEY, environment="production") as client:
        yield client


@pytest.fixture(scope="module")
def parsed(production_client: LandingAIADE) -> V2ParseResponse:
    """One real V2 parse, shared by the parse-shape and ground tests to hold spend down.

    The inline-grounding and parse-job tests issue their own calls (different options /
    endpoint), so they do not use this.
    """
    return production_client.v2.parse(document=SAMPLE_PDF)


def test_extract_sync(production_client: LandingAIADE) -> None:
    res = production_client.v2.extract(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
    assert isinstance(res, V2ExtractResult)
    assert isinstance(res.extraction, dict)
    assert res.extraction
    # `version` was renamed to `model_version` upstream; the current gateway populates it.
    assert res.metadata.model_version


def test_extract_jobs(production_client: LandingAIADE) -> None:
    job = production_client.v2.extract_jobs.create(schema=RevenueSchema, markdown=SAMPLE_MARKDOWN)
    done = production_client.v2.extract_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2ExtractResult)
    # Inline job: the metadata rides on `result.metadata`; the top-level `Job.metadata`
    # receipt is only populated for `output_save_url` deliveries.
    assert done.metadata is None
    assert done.result.metadata.model_version


def test_parse_sync(parsed: V2ParseResponse) -> None:
    assert isinstance(parsed, V2ParseResponse)
    assert isinstance(parsed.markdown, str)
    assert parsed.markdown


def test_parse_sync_inline_grounding_and_metadata(production_client: LandingAIADE) -> None:
    # Exercise the current parse surface: `inline_markdown` option, per-node spatial
    # `grounding` ({page, range, box}) inline on `structure`, and the renamed
    # `output_markdown_chars` / `range_units` metadata fields.
    resp = production_client.v2.parse(document=SAMPLE_PDF, options={"inline_markdown": True})
    assert isinstance(resp, V2ParseResponse)
    assert resp.structure is not None and resp.structure.children
    page = resp.structure.children[0]
    assert page.grounding is not None and page.grounding.range is not None
    assert page.grounding.box is not None
    assert resp.metadata is not None
    assert resp.metadata.range_units == "unicode_codepoints"
    assert resp.metadata.output_markdown_chars is not None


def test_ground_sync(production_client: LandingAIADE, parsed: V2ParseResponse) -> None:
    # Ground is a stateless join: extract against the parsed markdown, then ground the
    # extraction back onto the parse structure the markdown came from.
    assert parsed.structure is not None
    extracted = production_client.v2.extract(schema=RevenueSchema, markdown=parsed.markdown or "")
    grounded = production_client.v2.ground(
        extraction_metadata=extracted.extraction_metadata,
        structure=parsed.structure,
    )
    assert isinstance(grounded, V2GroundResult)
    assert isinstance(grounded.grounding, dict)
    assert grounded.metadata.job_id


def test_parse_jobs(production_client: LandingAIADE) -> None:
    job = production_client.v2.parse_jobs.create(document=SAMPLE_PDF)
    done = production_client.v2.parse_jobs.wait(job.job_id, timeout=300)
    assert done.status is JobStatus.COMPLETED
    # Assert the normalized job result, not just the terminal status, so this actually
    # covers the parse-job response contract (data -> V2ParseResponse).
    assert isinstance(done.result, V2ParseResponse)
    assert isinstance(done.result.markdown, str)
    assert done.result.markdown
    # Inline delivery: metadata rides on `result.metadata`, so the top-level `Job.metadata`
    # receipt (set only for `output_save_url` deliveries) is absent.
    assert done.metadata is None
    assert done.result.metadata is not None
