from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import respx
import pytest
from pydantic import Field, BaseModel

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import JobStatus, V2ExtractResult
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
EXTRACT_BODY: Dict[str, Any] = {
    "extraction": {"revenue": "1M"},
    "extraction_metadata": {"revenue": {"value": "1M", "spans": []}},
    "markdown": "# doc",
    "metadata": {"job_id": "e1", "version": "extract-1", "duration_ms": 5},
}


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


@respx.mock
def test_extract_sync_json_body_with_pydantic_schema() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    result = client.v2.extract(schema=Invoice, markdown="# doc")
    assert isinstance(result, V2ExtractResult) and result.metadata.version == "extract-1"
    req = json.loads(route.calls.last.request.content)
    assert req["schema"]["type"] == "object" and "revenue" in req["schema"]["properties"]
    assert req["markdown"] == "# doc"
    assert route.calls.last.request.headers["content-type"].startswith("application/json")


@respx.mock
def test_extract_sync_strict_option() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(schema={"type": "object", "properties": {}}, markdown_url="https://x/y.md", strict=True)
    req = json.loads(route.calls.last.request.content)
    assert req["options"]["strict"] is True
    assert req["markdown_url"] == "https://x/y.md"


@respx.mock
def test_extract_sync_grounding_option() -> None:
    # `grounding=False` skips the grounding stage server-side (every
    # `extraction_metadata` leaf comes back with `ranges: null`). It folds into
    # `options.grounding`, the second hand-written top-level shorthand on extract
    # alongside `strict` -- `bool()`, not truthiness, so `False` is SENT rather
    # than dropped, which is the only value anyone passes this for.
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(schema={"type": "object"}, markdown="# doc", grounding=False)
    req = json.loads(route.calls.last.request.content)
    assert req["options"] == {"grounding": False}


@respx.mock
def test_extract_options_carries_only_the_wired_shorthands() -> None:
    # `V2ExtractOptions` is `additionalProperties: false` upstream and carries
    # exactly two members, `strict` and `grounding`, both wired as top-level
    # shorthands. Pin the exact object rather than one key: the two fold into the
    # SAME nested object, so a regression that assigns `options` per shorthand
    # instead of merging would silently drop one, and any third key riding along
    # would be rejected by the gateway.
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(schema={"type": "object"}, markdown="# doc", strict=False, grounding=False)
    req = json.loads(route.calls.last.request.content)
    assert req["options"] == {"strict": False, "grounding": False}


@respx.mock
def test_extract_omits_options_when_no_shorthand_given() -> None:
    # Neither shorthand given -> no `options` at all, so the server's own defaults
    # apply (`strict` false, `grounding` true). `None` means "not given" for both,
    # the same as omitting them: there is no null to send here -- `options` is
    # where a caller would express "default", and leaving it out IS that.
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(schema={"type": "object"}, markdown="# doc", strict=None, grounding=None)
    req = json.loads(route.calls.last.request.content)
    assert "options" not in req


@respx.mock
def test_extract_extra_body_options_replaces_the_shorthands() -> None:
    # `extra_body` merges at the top level only (`_merge_mappings` is a shallow
    # `{**a, **b}`), so an `options` supplied there replaces the one the shorthands
    # assembled rather than merging into it. Both shorthands are now keywords, so
    # nobody needs `extra_body` to reach `grounding` -- but the shallow merge is
    # still the documented behavior of `extra_body` everywhere, and a caller who
    # mixes the two gets the override, not a union.
    client = LandingAIADE(apikey=APIKEY)
    route = respx.post("https://api.ade.landing.ai/v2/extract").mock(
        return_value=httpx.Response(200, json=EXTRACT_BODY)
    )
    client.v2.extract(
        schema={"type": "object"},
        markdown="# doc",
        strict=True,
        grounding=True,
        extra_body={"options": {"grounding": False}},
    )
    req = json.loads(route.calls.last.request.content)
    assert req["options"] == {"grounding": False}


def test_extract_job_create_carries_both_shorthands(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same `additionalProperties: false` contract on the async job route, which
    # shares `_build_extract_body` with the sync one -- including the merge that
    # keeps both shorthands in one `options` object.
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, options: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "e1", "status": "pending"}

    monkeypatch.setattr(client.v2.extract_jobs, "_post", fake_post)
    client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x", strict=True, grounding=False)
    assert captured["body"]["options"] == {"strict": True, "grounding": False}


@respx.mock
def test_extract_requires_a_markdown_source() -> None:
    # api.md documents the contract: provide exactly one of markdown /
    # markdown_url. Omitting both used to send a sourceless
    # body and surface an opaque server 500; the SDK now fails fast client-side.
    # No route is registered: @respx.mock keeps this hermetic, so a regression in
    # the guard fails loudly on an unmocked request instead of hitting the network.
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError, match="exactly one"):
        client.v2.extract(schema={"type": "object"})


@respx.mock
def test_extract_rejects_multiple_markdown_sources() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError, match="exactly one"):
        client.v2.extract(schema={"type": "object"}, markdown="x", markdown_url="https://x/y.md")


@respx.mock
def test_extract_job_create_requires_a_markdown_source() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError, match="exactly one"):
        client.v2.extract_jobs.create(schema={"type": "object"})


@respx.mock
def test_extract_sync_504() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://api.ade.landing.ai/v2/extract").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.extract(schema={"type": "object"}, markdown="x")


@respx.mock
def test_extract_job_create_and_get() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(
            202, json={"job_id": "e1", "status": "pending", "created_at": "2026-01-01T00:00:00Z"}
        )
    )
    job = client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x", service_tier="priority")
    assert job.job_id == "e1" and job.status is JobStatus.PENDING

    respx.get("https://api.ade.landing.ai/v2/extract/jobs/e1").mock(
        return_value=httpx.Response(
            200,
            json={
                "job_id": "e1",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:09Z",
                "result": EXTRACT_BODY,
            },
        )
    )
    done = client.v2.extract_jobs.get("e1")
    assert done.status is JobStatus.COMPLETED
    assert isinstance(done.result, V2ExtractResult)
    assert done.result.metadata.version == "extract-1"


def test_extract_job_create_sends_service_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    # The async job create body must carry `service_tier` (renamed from the
    # old `priority` field per the V2 spec).
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, options: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "e1", "status": "pending"}

    monkeypatch.setattr(client.v2.extract_jobs, "_post", fake_post)
    client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x", service_tier="priority")
    assert captured["body"]["service_tier"] == "priority"
    assert "priority" not in captured["body"]


@respx.mock
def test_extract_sync_parses_billing_metadata() -> None:
    # The V2 spec adds a `billing` object to the response metadata.
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(EXTRACT_BODY)
    metadata: Dict[str, Any] = dict(EXTRACT_BODY["metadata"])
    metadata["billing"] = {"service_tier": "priority", "total_credits": 12.5}
    body["metadata"] = metadata
    respx.post("https://api.ade.landing.ai/v2/extract").mock(return_value=httpx.Response(200, json=body))
    result = client.v2.extract(schema={"type": "object"}, markdown="x")
    assert result.metadata.billing is not None
    assert result.metadata.billing.service_tier == "priority"
    assert result.metadata.billing.total_credits == 12.5


@respx.mock
def test_extract_sync_parses_char_counts_and_warnings() -> None:
    # `input_markdown_chars`/`output_extraction_chars` moved onto `metadata`
    # (from `billing`) upstream, and `schema_violation_error`/`warnings` were
    # added to the result.
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(EXTRACT_BODY)
    metadata: Dict[str, Any] = dict(EXTRACT_BODY["metadata"])
    metadata["input_markdown_chars"] = 42
    metadata["output_extraction_chars"] = 7
    body["metadata"] = metadata
    body["schema_violation_error"] = "unsupported field skipped"
    body["warnings"] = [{"code": "partial", "message": "heads up"}]
    respx.post("https://api.ade.landing.ai/v2/extract").mock(return_value=httpx.Response(200, json=body))
    result = client.v2.extract(schema={"type": "object"}, markdown="x")
    assert result.metadata.input_markdown_chars == 42
    assert result.metadata.output_extraction_chars == 7
    assert result.schema_violation_error == "unsupported field skipped"
    assert result.warnings is not None and result.warnings[0]["code"] == "partial"


def test_extract_job_create_sends_output_save_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # The async job create body carries `output_save_url` (async jobs only).
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, options: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "e1", "status": "pending"}

    monkeypatch.setattr(client.v2.extract_jobs, "_post", fake_post)
    client.v2.extract_jobs.create(schema={"type": "object"}, markdown="x", output_save_url="https://example.com/put")
    assert captured["body"]["output_save_url"] == "https://example.com/put"


@respx.mock
def test_extract_job_get_failed_maps_error_object() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/jobs/e2").mock(
        return_value=httpx.Response(
            200, json={"job_id": "e2", "status": "failed", "error": {"code": "internal_error", "message": "boom"}}
        )
    )
    job = client.v2.extract_jobs.get("e2")
    assert job.status is JobStatus.FAILED and job.error is not None and job.error.code == "internal_error"


@respx.mock
def test_extract_job_wait_raise_on_failure() -> None:
    from landingai_ade.lib.v2_errors import JobFailedError

    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/jobs/e3").mock(
        return_value=httpx.Response(
            200, json={"job_id": "e3", "status": "failed", "error": {"code": "x", "message": "no"}}
        )
    )
    with pytest.raises(JobFailedError):
        client.v2.extract_jobs.wait("e3", timeout=5, poll_interval=0.01, raise_on_failure=True, _monotonic=lambda: 0.0)


@respx.mock
def test_extract_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.extract_jobs.get("")


@respx.mock
def test_extract_job_list_status_none_omits_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.extract_jobs.list(status=None)
    assert "status" not in route.calls.last.request.url.params


def test_extract_job_list_status_none_excluded_from_query_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test at the query-dict level: the underlying querystring
    # encoder happens to drop `None`-valued params when serializing to a URL,
    # which would mask this bug in an end-to-end/respx assertion. Capture the
    # dict handed to `options["params"]` directly so a regression is caught
    # even before it reaches that encoder.
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_get(path: str, *, cast_to: Any, options: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["params"] = dict(options or {}).get("params", {})
        return {"jobs": [], "has_more": False}

    monkeypatch.setattr(client.v2.extract_jobs, "_get", fake_get)

    client.v2.extract_jobs.list(status=None)
    assert "status" not in captured["params"]

    client.v2.extract_jobs.list(status="completed")
    assert captured["params"].get("status") == "completed"


@respx.mock
def test_extract_job_list_status_given_includes_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.extract_jobs.list(status="completed")
    assert route.calls.last.request.url.params["status"] == "completed"


@respx.mock
def test_extract_job_list_sends_page_size_as_camel_case() -> None:
    # The spec renamed the list query parameter `page_size` -> `pageSize`. The
    # `page_size` keyword is unchanged (surface-locked); only the wire name moved,
    # so the old snake_case key must not be sent alongside it.
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.extract_jobs.list(page=2, page_size=25)
    params = route.calls.last.request.url.params
    assert params["pageSize"] == "25"
    assert params["page"] == "2"
    assert "page_size" not in params


@respx.mock
def test_extract_job_list_normalizes_cancelled_status() -> None:
    # `cancelled` was added to the /v2/extract/jobs list status enum. It maps to
    # `JobStatus.CANCELLED` and counts as terminal.
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [{"job_id": "e9", "status": "cancelled", "completed_at": "2026-01-01T00:00:09Z"}],
                "has_more": False,
            },
        )
    )
    jobs = client.v2.extract_jobs.list()
    assert len(jobs) == 1
    assert jobs[0].status is JobStatus.CANCELLED
    assert jobs[0].is_terminal is True


@respx.mock
def test_extract_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/extract/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [{"job_id": "e1", "status": "completed"}],
                "page": 0,
                "page_size": 10,
                "has_more": True,
            },
        )
    )
    jobs = client.v2.extract_jobs.list()
    assert len(jobs) == 1 and jobs[0].job_id == "e1" and jobs[0].status is JobStatus.COMPLETED
    assert jobs.has_more is True
    assert jobs.page == 0
    assert jobs.page_size == 10
