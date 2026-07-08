"""
Custom ADE SDK utilities and extensions.

This package contains custom functionality that extends the base Stainless-generated SDK.
"""

from .schema_utils import coerce_schema_to_dict, pydantic_to_json_schema

__all__ = [
    "pydantic_to_json_schema",
    "coerce_schema_to_dict",
]
