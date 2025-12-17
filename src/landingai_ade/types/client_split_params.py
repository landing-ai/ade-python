# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Iterable, Optional
from typing_extensions import Required, Annotated, TypedDict

from .._types import FileTypes
from .._utils import PropertyInfo

__all__ = ["ClientSplitParams", "SplitClass"]


class ClientSplitParams(TypedDict, total=False):
    split_class: Required[Iterable[SplitClass]]
    """List of split classification options/configuration.

    Can be provided as JSON string in form data.
    """

    markdown: Union[FileTypes, str, None]
    """The Markdown file or Markdown content to split."""

    markdown_url: Annotated[Optional[str], PropertyInfo(alias="markdownUrl")]
    """The URL to the Markdown file to split."""

    model: Optional[str]
    """Model version to use for split classification. Defaults to the latest version."""


class SplitClass(TypedDict, total=False):
    """Model for split classification option."""

    name: Required[str]
    """Name of the split classification type"""

    description: Optional[str]
    """Detailed description of what this split type represents"""

    identifier: Optional[str]
    """Identifier to partition/group the splits by"""
