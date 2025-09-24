# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, TypedDict

from .._types import FileTypes

__all__ = ["ClientParseParams"]


class ClientParseParams(TypedDict, total=False):
    document: Optional[FileTypes]
    """A file to be parsed.

    The file can be a PDF (50 pages max) or an image (50MB). See the list of
    supported file types here (https://docs.landing.ai/ade/ade-file-types). Either
    this parameter or the document_url parameter must be provided.
    """

    document_url: Optional[str]
    """The URL to the file to be parsed.

    The file can be a PDF (50 pages max) or an image (50MB). See the list of
    supported file types here (https://docs.landing.ai/ade/ade-file-types). Either
    this parameter or the document parameter must be provided.
    """

    model: Optional[str]
    """The version of the model to use for parsing."""

    split: Optional[Literal["page"]]
    """
    If you want to split documents into smaller sections, include the split
    parameter. Set the parameter to page to split documents at the page level. The
    splits object in the API output will contain a set of data for each page.
    """
