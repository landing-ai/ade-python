# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from typing_extensions import TypedDict

from .._types import FileTypes

__all__ = ["ClientSectionParams"]


class ClientSectionParams(TypedDict, total=False):
    guidelines: Optional[str]
    """Natural-language instructions to control hierarchy.

    Examples: 'Group by topic', 'Treat each numbered section as a top-level entry'.
    """

    markdown: Union[FileTypes, str, None]
    """Parsed markdown with reference anchors (<a id='...'></a>).

    This is the markdown field from a parse response.
    """

    markdown_url: Optional[str]
    """URL to fetch the markdown from."""

    model: Optional[str]
    """Section model version. Defaults to latest."""
