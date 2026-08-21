from __future__ import annotations

import json
from typing import Any, Dict

from landingai_ade._compat import PYDANTIC_V1, parse_obj
from landingai_ade.types.parse_response import ParseResponse


def _payload(*, class_key: str) -> Dict[str, Any]:
    return {
        "chunks": [
            {
                "id": "c1",
                "grounding": {"box": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}, "page": 0},
                "markdown": "hello",
                "type": "text",
            }
        ],
        "markdown": "hello",
        "metadata": {
            "credit_usage": 1.0,
            "duration_ms": 1,
            "filename": "doc.pdf",
            "job_id": "j1",
            "page_count": 1,
        },
        "splits": [
            {
                class_key: "full",
                "identifier": "full",
                "markdown": "hello",
                "pages": [0],
                "chunks": ["c1"],
            }
        ],
    }


def _validate(raw: str) -> ParseResponse:
    if PYDANTIC_V1:
        return parse_obj(ParseResponse, json.loads(raw))
    return ParseResponse.model_validate_json(raw)


def test_parse_response_dump_does_not_duplicate_class_keys() -> None:
    response = ParseResponse.construct(**_payload(class_key="class"))
    dumped = json.loads(response.model_dump_json(indent=2))
    split = dumped["splits"][0]
    assert not ("class" in split and "class_" in split)
    assert split.get("class") == "full" or split.get("class_") == "full"


def test_parse_response_validates_json_with_class_underscore() -> None:
    parsed = _validate(json.dumps(_payload(class_key="class_")))
    assert parsed.splits[0].class_ == "full"


def test_parse_response_round_trips_dumped_json() -> None:
    response = ParseResponse.construct(**_payload(class_key="class"))
    raw = response.model_dump_json(indent=2)
    parsed = _validate(raw)
    assert parsed.splits[0].class_ == "full"
