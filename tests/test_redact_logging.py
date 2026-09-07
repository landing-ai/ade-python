"""Debug-log redaction: binary payloads shortened, secrets removed.

`_build_request` logs the whole request-options payload at DEBUG. That payload
carries uploaded file bytes (megabytes of noise) and, since `/v2/parse` gained the
encrypted-PDF option, the document password -- both as its own form field and inside
the JSON-encoded `options` field. Neither may reach a log.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, cast

import httpx
import respx

from landingai_ade import LandingAIADE
from landingai_ade._base_client import _redact_for_logging

APIKEY = "My Apikey"
PARSE_BODY: Dict[str, Any] = {
    "markdown": "# Hello",
    "metadata": {"req_id": "r1", "job_id": "j1", "model_version": "dpt-3", "page_count": 1, "failed_pages": []},
}


def test_binary_payloads_are_shortened() -> None:
    payload = {"files": [("document", ("a.pdf", b"x" * 5_000, "application/pdf"))]}
    redacted = cast(Dict[str, Any], _redact_for_logging(payload))
    assert redacted["files"][0][1][1] == "<5000 bytes>"


def test_password_form_field_is_redacted() -> None:
    redacted = cast(Dict[str, Any], _redact_for_logging({"data": {"password": "hunter2", "model": "dpt-3-pro"}}))
    assert redacted["data"]["password"] == "<redacted>"
    assert redacted["data"]["model"] == "dpt-3-pro"


def test_password_inside_the_json_encoded_options_field_is_redacted() -> None:
    # The one the form-field rule alone would miss: `options` is a JSON STRING, so the
    # password sits inside a value, not under a key of the payload being logged.
    payload = {"data": {"options": json.dumps({"pages": [1], "password": "hunter2"})}}
    redacted = cast(Dict[str, Any], _redact_for_logging(payload))
    reparsed = cast(Dict[str, Any], json.loads(redacted["data"]["options"]))
    assert reparsed["password"] == "<redacted>"
    assert reparsed["pages"] == [1]


def test_key_matching_is_case_insensitive() -> None:
    redacted = cast(Dict[str, Any], _redact_for_logging({"Password": "hunter2"}))
    assert redacted["Password"] == "<redacted>"


def test_a_payload_with_nothing_to_redact_is_returned_unchanged() -> None:
    # Identity, not just equality: the common case must not build a copy.
    payload = {"data": {"model": "dpt-3-pro", "options": json.dumps({"pages": [1]})}}
    assert _redact_for_logging(payload) is payload


def test_a_json_escaped_key_is_still_redacted() -> None:
    # A lexical search for "password" misses this: the key decodes to `password` only
    # after parsing, which is why the pre-check parses JSON-looking strings.
    payload = {"data": {"options": '{"pass\\u0077ord": "hunter2"}'}}
    redacted = cast(Dict[str, Any], _redact_for_logging(payload))
    assert "hunter2" not in json.dumps(redacted)
    assert cast(Dict[str, Any], json.loads(redacted["data"]["options"]))["password"] == "<redacted>"


def test_a_non_json_string_is_left_alone() -> None:
    # Rewriting only well-formed JSON keeps the redactor from mangling prose. A string
    # that merely mentions the word is not a secret this SDK serialized.
    text = "the password prompt appeared"
    assert _redact_for_logging({"note": text}) == {"note": text}


@respx.mock
def test_the_password_never_reaches_the_debug_log(caplog: Any) -> None:
    # End to end through the real logging call site, the way a user hits it.
    client = LandingAIADE(apikey=APIKEY, environment="production")
    respx.post("https://api.ade.landing.ai/v2/parse").mock(return_value=httpx.Response(200, json=PARSE_BODY))
    with caplog.at_level(logging.DEBUG, logger="landingai_ade"):
        client.v2.parse(document=b"%PDF", password="hunter2")
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "Request options" in logged  # the log line we care about actually ran
    assert "hunter2" not in logged
    assert "<redacted>" in logged
