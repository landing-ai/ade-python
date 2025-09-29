# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, TypedDict

from .._types import FileTypes

__all__ = ["ClientExtractParams"]


class ClientExtractParams(TypedDict, total=False):
    schema: Required[str]
    """JSON schema for field extraction.

    This schema determines what key-values pairs are extracted from the Markdown.
    The schema must be a valid JSON object and will be validated before processing
    the document.
    """

    markdown: Optional[FileTypes]
    """The Markdown file or Markdown content to extract data from."""

    markdown_url: Optional[str]
    """The URL to the Markdown file to extract data from."""

    model: Optional[Literal["extract-20250630", "extract-20250930"]]
    """The version of the model to use for extraction."""
