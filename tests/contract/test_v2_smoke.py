from __future__ import annotations

import os
from typing import List, Tuple, Iterator, Optional
from pathlib import Path

import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import (
    JobStatus,
    V2GroundResult,
    V2ParseElement,
    V2ExtractResult,
    V2ParseResponse,
    V2ParseNodeGrounding,
)

pytestmark = pytest.mark.contract

STAGING_KEY = os.environ.get("LANDINGAI_ADE_STAGING_APIKEY")

# A tiny self-contained markdown document so extract/files can run without any file.
SAMPLE_MARKDOWN = "# Acme Inc. — Q1 Report\n\nTotal revenue for the quarter was **$1,250,000**.\n"

# These tests are a merge gate against a LIVE environment, so they have to fail fast and
# legibly. The SDK ships an 8-minute timeout with 2 retries — sensible for a real caller
# parsing a 500-page scan, wrong for a gate: an endpoint staging accepts but never answers
# then burns ~24 minutes and reads as "the suite is slow" rather than "staging never
# replied". Cap one request at 45s and don't retry, so a hang surfaces as an
# `APITimeoutError` naming the route.
#
# Trade-off: a transient 429/5xx now fails the run instead of being retried away. That is
# the intended bias — a retry cannot rescue a genuinely dead upstream, it only hides it —
# and the whole suite is ~100s, so re-running the job is cheap.
CONTRACT_TIMEOUT = 45.0

# Budget for the `.wait()` job polls below. `poll_until_terminal` (_base.py) calls `get_job()`
# and only THEN checks the deadline, so a poll starting just under it still costs a full
# CONTRACT_TIMEOUT on top: worst case per job test is create + deadline + one in-flight poll.
# With these values that is 45+60+45=150s (extract) and 45+120+45=210s (parse), which keeps the
# whole file inside the 15-minute `contract-tests` job cap even if staging stops answering
# entirely. Parse gets the larger deadline because it processes the 2-page sample PDF rather
# than a few hundred bytes of markdown. Raise either only against that cap.
EXTRACT_JOB_WAIT = 60.0
PARSE_JOB_WAIT = 120.0


class RevenueSchema(BaseModel):
    """Demonstrates passing a pydantic model as the extract schema."""

    revenue: str = Field(description="The total revenue figure, verbatim")
    company: str = Field(description="The company name")


@pytest.fixture()
def staging_client() -> Iterator[LandingAIADE]:
    if not STAGING_KEY:
        pytest.skip("LANDINGAI_ADE_STAGING_APIKEY not set")
    # Context-managed so the underlying HTTP client is closed in teardown (no socket leak).
    with LandingAIADE(
        apikey=STAGING_KEY,
        environment="staging",
        timeout=CONTRACT_TIMEOUT,
        max_retries=0,
    ) as client:
        yield client


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
    done = staging_client.v2.extract_jobs.wait(job.job_id, timeout=EXTRACT_JOB_WAIT)
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2ExtractResult)
    # This inline job carries its metadata on `result.metadata`; the top-level
    # `Job.metadata` receipt is only populated for `output_save_url` deliveries.
    assert done.metadata is None
    assert done.result.metadata.model_version


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


def _split_confidences(resp: V2ParseResponse) -> Tuple[List[float], List[float]]:
    # Walk the structure tree once and return the confidence values that are actually
    # set, split by where they were found: node-level `grounding` (page and element
    # nodes) vs word `atomic_grounding` segments. Returning the values rather than
    # asserting inline is what lets a caller assert a count -- an in-tree `if
    # confidence is not None` check passes vacuously when nothing carries a score,
    # which is exactly how this contract went unverified.
    node: List[float] = []
    atomic: List[float] = []

    def _node(grounding: Optional[V2ParseNodeGrounding]) -> None:
        if grounding is not None and grounding.confidence is not None:
            node.append(grounding.confidence)

    def _walk(elements: List[V2ParseElement]) -> None:
        for el in elements:
            _node(el.grounding)
            for seg in el.atomic_grounding or []:
                if seg.confidence is not None:
                    atomic.append(seg.confidence)
            _walk(el.children or [])

    assert resp.structure is not None
    for page in resp.structure.children:
        _node(page.grounding)
        _walk(page.children)
    return node, atomic


def test_parse_atomic_grounding_confidence(staging_client: LandingAIADE) -> None:
    # `confidence` is an optional `[0, 1]` probability that lives ONLY on word
    # `atomic_grounding` segments. Node-level `grounding` -- element, `table_cell`,
    # `table`, page -- never carries it on ANY model: the weakest-link roll-up the
    # gateway used to do was removed upstream. That half of the contract is
    # model-independent, so it is asserted here on the default model.
    #
    # The positive half (words actually carrying scores) needs a word-granularity
    # model and lives in the test below -- on the default `dpt-3-pro` nothing carries
    # a score at all, so the `atomic` range check here is vacuous by design.
    pdf = Path(__file__).parent / "sample.pdf"
    resp = staging_client.v2.parse(document=pdf, options={"atomic_grounding": True})
    assert isinstance(resp, V2ParseResponse)

    node, atomic = _split_confidences(resp)
    assert node == [], f"node-level grounding must not carry confidence, got {node[:3]}"
    assert all(0.0 <= c <= 1.0 for c in atomic)


def test_parse_atomic_grounding_confidence_word_granularity(staging_client: LandingAIADE) -> None:
    # The positive half of the confidence contract: pin a word-granularity model so
    # the path is actually exercised.
    #
    # Split out rather than folded into the test above because pinning costs
    # something: `dpt-3-verity` is GPU-gated on staging, so a cluster booked without
    # a GPU hangs this request until CONTRACT_TIMEOUT. Isolated, that infra case reds
    # one obviously-named test instead of the general grounding contract. (Latency is
    # not the concern -- verity and pro both parse the sample in ~12s.)
    #
    # `>= 1` scored segment, not "every segment": the model legitimately leaves some
    # unscored -- text it wrote rather than read, cells whose words it could not
    # locate in the rendered text -- 297 of 299 on this sample.
    pdf = Path(__file__).parent / "sample.pdf"
    resp = staging_client.v2.parse(document=pdf, model="dpt-3-verity", options={"atomic_grounding": True})
    assert isinstance(resp, V2ParseResponse)

    node, atomic = _split_confidences(resp)
    assert node == [], f"node-level grounding must not carry confidence, got {node[:3]}"
    assert atomic, "word-granularity parse returned no scored atomic segments"
    assert all(0.0 <= c <= 1.0 for c in atomic)


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


def test_parse_jobs(staging_client: LandingAIADE) -> None:
    pdf = Path(__file__).parent / "sample.pdf"
    job = staging_client.v2.parse_jobs.create(document=pdf)
    done = staging_client.v2.parse_jobs.wait(job.job_id, timeout=PARSE_JOB_WAIT)
    assert done.status is JobStatus.COMPLETED
    # Assert the normalized job result, not just the terminal status, so this test
    # actually covers the parse-job response contract (data -> V2ParseResponse).
    assert isinstance(done.result, V2ParseResponse)
    assert isinstance(done.result.markdown, str)
    assert done.result.markdown
    # Inline delivery: the metadata rides on `result.metadata`, so the top-level
    # `Job.metadata` receipt (set only for `output_save_url` deliveries) is absent.
    assert done.metadata is None
    assert done.result.metadata is not None
