# tests/test_v2_schema.py
from __future__ import annotations

import pytest
from pydantic import Field, BaseModel

from landingai_ade.lib.schema_utils import coerce_schema_to_dict


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


def test_coerce_from_pydantic_model() -> None:
    out = coerce_schema_to_dict(Invoice)
    assert out["type"] == "object"
    assert "revenue" in out["properties"]
    assert "$defs" not in out  # refs resolved & stripped


def test_coerce_from_dict_passthrough() -> None:
    d = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert coerce_schema_to_dict(d) == d


def test_coerce_from_json_string() -> None:
    out = coerce_schema_to_dict('{"type": "object", "properties": {}}')
    assert out == {"type": "object", "properties": {}}


def test_coerce_rejects_garbage() -> None:
    with pytest.raises((ValueError, TypeError)):
        coerce_schema_to_dict("not json")
    with pytest.raises(TypeError):
        coerce_schema_to_dict(123)  # type: ignore[arg-type]
