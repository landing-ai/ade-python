from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import httpx
import respx
import pytest

from landingai_ade import LandingAIADE
from landingai_ade.types.v2 import Job, JobStatus, V2ParseResponse
from landingai_ade.lib.v2_errors import V2SyncTimeoutError

APIKEY = "My Apikey"
PARSE_BODY: Dict[str, Any] = {
    "markdown": "# Hello",
    "structure": {"type": "document", "children": []},
    "grounding": {"type": "document", "children": []},
    "metadata": {"req_id": "r1", "job_id": "j1", "model_version": "dpt-3", "page_count": 1, "failed_pages": []},
}

# A fully-populated ParseResponse mirroring the typed gateway schema: a document
# with one page holding a text element and a table (with a table_cell child), plus
# the parallel grounding tree carrying boxes/parts.
RICH_PARSE_BODY: Dict[str, Any] = {
    "markdown": "# Invoice",
    "metadata": {"req_id": "r1", "job_id": "j1", "model_version": "dpt-3", "page_count": 1, "failed_pages": []},
    "structure": {
        "type": "document",
        "children": [
            {
                "type": "page",
                "page": 0,
                "span": [0, 100],
                "width": 800,
                "height": 1000,
                "dpi": 150,
                "status": "ok",
                "children": [
                    {"type": "text", "id": "e1", "span": [0, 20]},
                    {
                        "type": "table",
                        "id": "e2",
                        "span": [20, 80],
                        "children": [
                            {
                                "type": "table_cell",
                                "id": "e3",
                                "span": [25, 30],
                                "row": 0,
                                "col": 1,
                                "colspan": 2,
                                "rowspan": 1,
                            },
                        ],
                    },
                ],
            }
        ],
    },
    "grounding": {
        "type": "document",
        "children": [
            {
                "type": "page",
                "page": 0,
                "span": [0, 100],
                "children": [
                    {
                        "type": "text",
                        "id": "e1",
                        "span": [0, 20],
                        "box": [10, 10, 200, 30],
                        "parts": [{"span": [0, 20], "box": [10, 10, 200, 30]}],
                    },
                    {
                        "type": "table",
                        "id": "e2",
                        "span": [20, 80],
                        "box": [10, 40, 400, 300],
                        "parts": [],
                        "children": [
                            {"type": "table_cell", "id": "e3", "span": [25, 30], "box": [12, 42, 100, 80], "parts": []},
                        ],
                    },
                ],
            }
        ],
    },
}


@respx.mock
def test_parse_sync_ok_routes_to_v2_and_sends_options_json() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    result = client.v2.parse(document=b"pdf", model="dpt-3-latest", options={"foo": "bar"})
    assert isinstance(result, V2ParseResponse)
    assert result.markdown == "# Hello"
    # options must be sent as a JSON-encoded string form field
    sent = route.calls.last.request.content
    assert b'{"foo": "bar"}' in sent or b'"foo"' in sent


@respx.mock
def test_parse_sync_omits_unset_fields_from_multipart_body() -> None:
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    client.v2.parse(document=b"pdf")
    # Unset `Omit`/`NotGiven` sentinels must never leak into the multipart
    # body as literal `"<...Omit object...>"` / `"NOT_GIVEN"` form fields.
    sent = route.calls.last.request.content
    assert b"Omit object" not in sent
    assert b"NOT_GIVEN" not in sent
    assert b"document_url" not in sent
    assert b"model" not in sent
    assert b"options" not in sent
    assert b"password" not in sent


@respx.mock
def test_parse_sync_omits_explicit_none_fields_from_multipart_body() -> None:
    # `is_given(None)` is True (it only filters the `omit`/`not_given` sentinels),
    # so an explicit `None` for an optional field must be dropped by hand -- it
    # must never leak into the multipart body as a literal "None"/"null" field.
    client = LandingAIADE(apikey=APIKEY, environment="production")
    route = respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    client.v2.parse(document=b"x", document_url=None, model=None, options=None, password=None)
    sent = route.calls.last.request.content
    assert b"document_url" not in sent
    assert b"model" not in sent
    assert b"options" not in sent
    assert b"password" not in sent
    assert b"null" not in sent
    assert b"None" not in sent


def test_parse_job_create_omits_explicit_none_extra_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test at the body-dict level (before multipart encoding, which
    # happens to drop bare `None` values and would otherwise mask this bug):
    # `output_save_url=None`/`service_tier=None` must not survive into the body
    # dict handed to the request layer.
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, files: Any = None, options: Any = None) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "p1", "status": "pending"}

    monkeypatch.setattr(client.v2.parse_jobs, "_post", fake_post)
    client.v2.parse_jobs.create(document=b"x", output_save_url=None, service_tier=None)
    assert "output_save_url" not in captured["body"]
    assert "service_tier" not in captured["body"]


def test_parse_job_create_sends_service_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    # The async job create body must carry `service_tier` (renamed from the
    # old `priority` field per the V2 spec).
    client = LandingAIADE(apikey=APIKEY)
    captured: Dict[str, Any] = {}

    def fake_post(path: str, *, cast_to: Any, body: Any = None, files: Any = None, options: Any = None) -> Any:  # noqa: ARG001
        captured["body"] = body
        return {"job_id": "p1", "status": "pending"}

    monkeypatch.setattr(client.v2.parse_jobs, "_post", fake_post)
    client.v2.parse_jobs.create(document=b"x", service_tier="priority")
    assert captured["body"]["service_tier"] == "priority"
    assert "priority" not in captured["body"]


@respx.mock
def test_parse_job_list_status_none_omits_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.parse_jobs.list(status=None)
    assert "status" not in route.calls.last.request.url.params


def test_parse_job_list_status_none_excluded_from_query_dict(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(client.v2.parse_jobs, "_get", fake_get)

    client.v2.parse_jobs.list(status=None)
    assert "status" not in captured["params"]

    client.v2.parse_jobs.list(status="completed")
    assert captured["params"].get("status") == "completed"


@respx.mock
def test_parse_job_list_status_given_includes_query_param() -> None:
    client = LandingAIADE(apikey=APIKEY)
    route = respx.get("https://api.ade.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [], "has_more": False})
    )
    client.v2.parse_jobs.list(status="completed")
    assert route.calls.last.request.url.params["status"] == "completed"


@respx.mock
def test_parse_sync_206_returns_response_with_failed_pages() -> None:
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(PARSE_BODY)
    metadata: Dict[str, Any] = dict(PARSE_BODY["metadata"])
    metadata["failed_pages"] = [3]
    body["metadata"] = metadata
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(206, json=body))
    result = client.v2.parse(document=b"pdf")
    assert result.metadata is not None and result.metadata.failed_pages == [3]


@respx.mock
def test_parse_sync_typed_structure_and_grounding() -> None:
    # The gateway now returns a typed ParseResponse; `structure` and `grounding`
    # deserialize into typed models with attribute access, not raw dicts.
    from landingai_ade.types.v2 import V2ParseGrounding, V2ParseStructure

    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=RICH_PARSE_BODY))
    result = client.v2.parse(document=b"pdf")

    assert isinstance(result.structure, V2ParseStructure)
    page = result.structure.children[0]
    assert page.page == 0 and page.width == 800 and page.dpi == 150 and page.status == "ok"
    text, table = page.children[0], page.children[1]
    assert text.type == "text" and text.id == "e1"
    assert table.type == "table" and table.children is not None
    cell = table.children[0]
    assert cell.type == "table_cell" and cell.row == 0 and cell.col == 1 and cell.colspan == 2

    assert isinstance(result.grounding, V2ParseGrounding)
    g_text = result.grounding.children[0].children[0]
    assert g_text.box == [10, 10, 200, 30]
    assert g_text.parts[0].span == [0, 20] and g_text.parts[0].box == [10, 10, 200, 30]
    g_table = result.grounding.children[0].children[1]
    assert g_table.children is not None and g_table.children[0].box == [12, 42, 100, 80]


@respx.mock
def test_parse_sync_tolerates_unknown_element_type_and_extra_keys() -> None:
    # Novel element `type` values and extra keys must not break deserialization
    # (`type` is a permissive str; BaseModel retains extra keys).
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = {
        "markdown": "x",
        "metadata": {"req_id": "r1", "job_id": "j1", "model_version": "m", "page_count": 1, "failed_pages": []},
        "structure": {
            "type": "document",
            "children": [
                {
                    "type": "page",
                    "page": 0,
                    "span": [0, 5],
                    "future_field": 7,
                    "children": [
                        {"type": "some_future_kind", "id": "e9", "span": [0, 5]},
                    ],
                },
            ],
        },
        "grounding": {"type": "document", "children": []},
    }
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=body))
    result = client.v2.parse(document=b"pdf")
    assert result.structure is not None
    page = result.structure.children[0]
    assert page.children[0].type == "some_future_kind"
    # the unknown `future_field` key must survive deserialization (extra="allow")
    assert page.to_dict()["future_field"] == 7


@respx.mock
def test_parse_sync_504_raises_v2_sync_timeout() -> None:
    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(504, json={"detail": "x"}))
    with pytest.raises(V2SyncTimeoutError):
        client.v2.parse(document=b"pdf")


@respx.mock
def test_parse_save_to_writes_file(tmp_path: Path) -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    client.v2.parse(document=b"pdf", save_to=str(tmp_path))
    written = list(tmp_path.glob("*.json"))
    assert written and json.loads(written[0].read_text())["markdown"] == "# Hello"


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_sync_ok() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY, environment="production")
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    result = await client.v2.parse(document=b"pdf")
    assert isinstance(result, V2ParseResponse)
    assert result.markdown == "# Hello"


@respx.mock
def test_parse_job_create_normalizes_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "p1", "status": "pending", "received_at": 1700000000})
    )
    job = client.v2.parse_jobs.create(document=b"pdf", service_tier="priority")
    assert isinstance(job, Job)
    assert job.job_id == "p1" and job.status is JobStatus.PENDING


@respx.mock
def test_parse_job_get_completed_has_typed_result() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/parse/jobs/p1").mock(
        return_value=httpx.Response(
            200,
            json={"job_id": "p1", "status": "completed", "created_at": 1700000000, "data": PARSE_BODY},
        )
    )
    job = client.v2.parse_jobs.get("p1")
    assert job.status is JobStatus.COMPLETED
    assert isinstance(job.result, V2ParseResponse)
    assert job.result.markdown == "# Hello"


@respx.mock
def test_parse_job_get_206_partial_returns_result() -> None:
    # Per the V2 spec, GET /v2/parse/jobs/{id} can return 206 (partial success).
    # 206 is a 2xx, so the base client treats it as success and the envelope is
    # normalized like any completed job.
    client = LandingAIADE(apikey=APIKEY)
    body: Dict[str, Any] = dict(PARSE_BODY)
    metadata: Dict[str, Any] = dict(PARSE_BODY["metadata"])
    metadata["failed_pages"] = [2]
    body["metadata"] = metadata
    respx.get("https://api.ade.landing.ai/v2/parse/jobs/p1").mock(
        return_value=httpx.Response(206, json={"job_id": "p1", "status": "completed", "data": body})
    )
    job = client.v2.parse_jobs.get("p1")
    assert job.status is JobStatus.COMPLETED
    assert isinstance(job.result, V2ParseResponse)
    assert job.result.metadata is not None and job.result.metadata.failed_pages == [2]


@respx.mock
def test_parse_job_get_404_raises_not_found() -> None:
    from landingai_ade import NotFoundError

    client = LandingAIADE(apikey=APIKEY, max_retries=0)
    respx.get("https://api.ade.landing.ai/v2/parse/jobs/missing").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(NotFoundError):
        client.v2.parse_jobs.get("missing")


@respx.mock
def test_parse_job_get_empty_job_id_raises() -> None:
    client = LandingAIADE(apikey=APIKEY)
    with pytest.raises(ValueError):
        client.v2.parse_jobs.get("")


@respx.mock
def test_parse_job_list_carries_envelope() -> None:
    client = LandingAIADE(apikey=APIKEY)
    respx.get("https://api.ade.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(
            200,
            json={"jobs": [{"job_id": "p1", "status": "completed"}], "org_id": "o1", "has_more": True},
        )
    )
    jobs = client.v2.parse_jobs.list(page=0)
    assert len(jobs) == 1 and jobs[0].job_id == "p1"
    assert jobs.has_more is True and jobs.org_id == "o1"


@respx.mock
def test_parse_job_wait_polls_until_completed() -> None:
    client = LandingAIADE(apikey=APIKEY)
    responses = [
        httpx.Response(200, json={"job_id": "p1", "status": "processing", "progress": 0.5}),
        httpx.Response(200, json={"job_id": "p1", "status": "completed", "data": PARSE_BODY}),
    ]
    respx.get("https://api.ade.landing.ai/v2/parse/jobs/p1").mock(side_effect=responses)
    # inject fake clock so no real time passes
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    job = client.v2.parse_jobs.wait("p1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks))
    assert job.status is JobStatus.COMPLETED


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_job_create_and_get() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    respx.post("https://api.ade.landing.ai/v2/parse/jobs").mock(
        return_value=httpx.Response(202, json={"job_id": "p1", "status": "pending"})
    )
    respx.get("https://api.ade.landing.ai/v2/parse/jobs/p1").mock(
        return_value=httpx.Response(200, json={"job_id": "p1", "status": "completed", "data": PARSE_BODY})
    )
    created = await client.v2.parse_jobs.create(document=b"pdf")
    assert created.status is JobStatus.PENDING
    fetched = await client.v2.parse_jobs.get("p1")
    assert fetched.status is JobStatus.COMPLETED
    assert isinstance(fetched.result, V2ParseResponse)
    assert fetched.result.markdown == "# Hello"


@respx.mock
@pytest.mark.asyncio
async def test_async_parse_job_wait_polls_until_completed() -> None:
    from landingai_ade import AsyncLandingAIADE

    client = AsyncLandingAIADE(apikey=APIKEY)
    responses = [
        httpx.Response(200, json={"job_id": "p1", "status": "processing", "progress": 0.5}),
        httpx.Response(200, json={"job_id": "p1", "status": "completed", "data": PARSE_BODY}),
    ]
    respx.get("https://api.ade.landing.ai/v2/parse/jobs/p1").mock(side_effect=responses)
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])
    job = await client.v2.parse_jobs.wait("p1", timeout=30, poll_interval=0.01, _monotonic=lambda: next(ticks))
    assert job.status is JobStatus.COMPLETED
