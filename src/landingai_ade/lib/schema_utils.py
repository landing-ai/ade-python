"""
Schema utilities for the ADE SDK.

This module provides utility functions for converting Pydantic models to JSON schemas
that can be used with the ADE API endpoints.
"""

import copy
import json
from typing import Any, Dict, Type

from pydantic import BaseModel


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
    if not hasattr(model, "model_json_schema"):
        raise TypeError("model must be a Pydantic BaseModel subclass")

    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    schema = _resolve_refs(schema, defs)
    return json.dumps(schema)
