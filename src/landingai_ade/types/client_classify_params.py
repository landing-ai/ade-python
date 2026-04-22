# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable, Optional
from typing_extensions import Required, TypedDict

from .._types import FileTypes

__all__ = ["ClientClassifyParams", "Class"]


class ClientClassifyParams(TypedDict, total=False):
    classes: Required[Iterable[Class]]
    """The possible classes that can be assigned to pages in the document.

    Each entry is an object with a `class` name and an optional `description`. Only
    one class is assigned per page; unclassifiable pages receive 'unknown'. Can be
    provided as a JSON string in form data.
    """

    document: Optional[FileTypes]
    """A file to be classified.

    Either this parameter or the `document_url` parameter must be provided.
    """

    document_url: Optional[str]
    """The URL of the document to be classified.

    Either this parameter or the `document` parameter must be provided.
    """

    model: Optional[str]
    """Classification model version. Defaults to the latest."""


_ClassReservedKeywords = TypedDict(
    "_ClassReservedKeywords",
    {
        "class": str,
    },
    total=False,
)


class Class(_ClassReservedKeywords, total=False):
    """A single classification option: a class name plus optional description."""

    description: Optional[str]
    """Detailed description of what this class represents."""
