# tests/test_v2_schema.py
from __future__ import annotations

from typing import Any, Dict, List, cast

import pytest
from pydantic import Field, BaseModel

from landingai_ade.lib.schema_utils import coerce_schema_to_dict


class Invoice(BaseModel):
    revenue: str = Field(description="Q1 revenue")


class LineItem(BaseModel):
    sku: str = Field(description="Item SKU")
    quantity: int = Field(description="Quantity ordered")


class PurchaseOrder(BaseModel):
    """Nested model: has a sub-model field, so its schema gets a
    `definitions`/`$defs` section that must be resolved."""

    order_id: str = Field(description="Order identifier")
    item: LineItem = Field(description="The line item")


def _find_all_keys(obj: Any, key: str) -> List[Any]:
    """Recursively collect all values for `key` anywhere in a nested dict/list."""
    found: List[Any] = []
    if isinstance(obj, dict):
        for k, v in cast(Dict[Any, Any], obj).items():
            if k == key:
                found.append(v)
            found.extend(_find_all_keys(v, key))
    elif isinstance(obj, list):
        for item in cast(List[Any], obj):
            found.extend(_find_all_keys(item, key))
    return found


def test_coerce_nested_model_is_idempotent_and_fully_resolved() -> None:
    """Regression test: pydantic v1's `.schema()` is memoized per class and
    returns the SAME shared mutable dict every call. If `coerce_schema_to_dict`
    (via `_model_json_schema`) mutates that dict in place instead of deep-copying
    it first, the first call strips "definitions"/"$defs" from the cached schema,
    and any subsequent call on the same class blows up with a KeyError in
    `_resolve_refs` (or leaves an unresolved "$ref"). Calling this twice on the
    same class is the intended usage pattern (e.g. reusing a schema class across
    multiple `client.v2.extract` / `extract_jobs.create` calls), so both calls
    must succeed and be fully resolved.
    """
    for _ in range(2):
        out = coerce_schema_to_dict(PurchaseOrder)
        assert out["type"] == "object"
        assert "order_id" in out["properties"]
        assert "item" in out["properties"]

        # No dangling $ref anywhere (nested sub-schema must be fully inlined),
        # and the nested model's own fields must actually be present.
        assert _find_all_keys(out, "$ref") == []
        all_property_maps = _find_all_keys(out, "properties")
        assert any("sku" in props for props in all_property_maps)
        assert any("quantity" in props for props in all_property_maps)

        # No leftover definitions/$defs anywhere in the resolved output.
        assert "$defs" not in out
        assert "definitions" not in out


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
