# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from typing_extensions import TypedDict

from .._types import FileTypes, SequenceNotStr

__all__ = ["ClientExtractBuildSchemaParams"]


class ClientExtractBuildSchemaParams(TypedDict, total=False):
    markdown_urls: Optional[SequenceNotStr[str]]
    """URLs to Markdown files to analyze for schema generation."""

    markdowns: Optional[SequenceNotStr[Union[FileTypes, str]]]
    """Markdown files or inline content strings to analyze for schema generation.

    Multiple documents can be provided for better schema coverage.
    """

    model: Optional[str]
    """The version of the model to use for schema generation.

    Use `extract-latest` to use the latest version.
    """

    prompt: Optional[str]
    """Instructions for how to generate or modify the schema."""

    schema: Optional[str]
    """Existing JSON schema to iterate on or refine."""
