# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from typing_extensions import Required, TypedDict

from .._types import FileTypes

__all__ = ["ClientExtractParams"]


class ClientExtractParams(TypedDict, total=False):
    schema: Required[str]
    """JSON schema for field extraction.

    This schema determines what key-values pairs are extracted from the Markdown.
    The schema must be a valid JSON object and will be validated before processing
    the document.
    """

    markdown: Union[FileTypes, str, None]
    """The Markdown file or Markdown content to extract data from."""

    markdown_url: Optional[str]
    """The URL to the Markdown file to extract data from."""

    model: Optional[str]
    """The version of the model to use for extraction.

    Use `extract-latest` to use the latest version.
    """
