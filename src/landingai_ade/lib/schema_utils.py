"""
Schema utilities for the ADE SDK.

This module provides utility functions for converting Pydantic models to JSON schemas
that can be used with the ADE API endpoints.
"""

import copy
import json
from typing import Any, Dict, Type, Union, Mapping, cast

from pydantic import BaseModel

from .._compat import PYDANTIC_V1


def _model_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Return a model's JSON schema, keyed the same way regardless of pydantic major.

    Pydantic v2 exposes `model_json_schema()` and nests shared definitions under
    `$defs`; pydantic v1 only has `.schema()` and nests them under `definitions`.
    We normalize to `$defs` here so callers (and `_resolve_refs`) don't need to
    know which pydantic major produced the schema.
    """
    if PYDANTIC_V1:
        schema = model.schema()  # type: ignore[attr-defined]
        if "definitions" in schema:
            schema["$defs"] = schema.pop("definitions")
        return schema
    return model.model_json_schema()


def _resolve_refs(obj: Any, defs: Dict[str, Any]) -> Any:
    """
    Resolve JSON Schema $refs to create a flat schema.

    This function recursively resolves all $ref references in a JSON schema
    by replacing them with their definitions from the $defs section.

    Args:
        obj: The schema object (or part of it) to process
        defs: Dictionary of schema definitions

    Returns:
        The schema with all $refs resolved
    """
    if isinstance(obj, dict):
        if "$ref" in obj and isinstance(obj["$ref"], str):
            ref_name = obj["$ref"].split("/")[-1]
            return _resolve_refs(copy.deepcopy(defs[ref_name]), defs)
        return {k: _resolve_refs(v, defs) for k, v in obj.items()}  # type: ignore[misc]
    elif isinstance(obj, list):
        return [_resolve_refs(item, defs) for item in obj]  # type: ignore[misc]
    return obj


def pydantic_to_json_schema(model: Type[BaseModel]) -> str:
    """
    Convert a Pydantic model to a JSON schema string.

    This utility function takes a Pydantic BaseModel class and converts it to a
    JSON schema string with all $refs resolved, suitable for use with the ADE API.

    Args:
        model: A Pydantic BaseModel class defining the schema

    Returns:
        JSON string representation of the schema

    Raises:
        TypeError: If model is not a Pydantic BaseModel subclass

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from landingai_ade.lib.schema_utils import pydantic_to_json_schema
        >>>
        >>> class Person(BaseModel):
        ...     name: str = Field(description="Person's name")
        ...     age: int = Field(description="Person's age")
        >>> schema_json = pydantic_to_json_schema(Person)
        >>> # Now use schema_json with the SDK:
        >>> # client.extract(schema=schema_json, markdown="...")
    """
    # The type annotation already ensures model is Type[BaseModel]
    # but we'll do a runtime check for safety
    if not (
        isinstance(model, type)  # pyright: ignore[reportUnnecessaryIsInstance]
        and issubclass(model, BaseModel)  # pyright: ignore[reportUnnecessaryIsInstance]
    ):
        raise TypeError("model must be a Pydantic BaseModel subclass")

    schema = _model_json_schema(model)
    defs = schema.pop("$defs", {})
    schema = _resolve_refs(schema, defs)
    return json.dumps(schema)


def pydantic_to_schema_dict(model: Type[BaseModel]) -> Dict[str, Any]:
    """Like `pydantic_to_json_schema` but returns a dict with $refs resolved."""
    if not (
        isinstance(model, type)  # pyright: ignore[reportUnnecessaryIsInstance]
        and issubclass(model, BaseModel)  # pyright: ignore[reportUnnecessaryIsInstance]
    ):
        raise TypeError("model must be a Pydantic BaseModel subclass")
    schema = _model_json_schema(model)
    defs = schema.pop("$defs", {})
    return cast(Dict[str, Any], _resolve_refs(schema, defs))


def coerce_schema_to_dict(schema: Union[str, Mapping[str, Any], Type[BaseModel]]) -> Dict[str, Any]:
    """Accept a pydantic model class, a dict, or a JSON string; return a JSON-Schema dict.

    The V2 extract endpoint takes `schema` as a JSON object in the request body.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
        return pydantic_to_schema_dict(schema)
    if isinstance(schema, Mapping):
        return dict(schema)
    if isinstance(schema, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        parsed: Any = json.loads(schema)  # raises ValueError on bad JSON
        if not isinstance(parsed, dict):
            raise TypeError("schema JSON string must decode to an object")
        return cast(Dict[str, Any], parsed)
    raise TypeError(f"Unsupported schema type: {type(schema)!r}")
