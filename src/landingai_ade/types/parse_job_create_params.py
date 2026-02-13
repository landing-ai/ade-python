# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, TypedDict

from .._types import FileTypes

__all__ = ["ParseJobCreateParams"]


class ParseJobCreateParams(TypedDict, total=False):
    document: Optional[FileTypes]
    """A file to be parsed.

    The file can be a PDF or an image. See the list of supported file types here:
    https://docs.landing.ai/ade/ade-file-types. Either this parameter or the
    `document_url` parameter must be provided.
    """

    document_url: Optional[str]
    """The URL to the file to be parsed.

    The file can be a PDF or an image. See the list of supported file types here:
    https://docs.landing.ai/ade/ade-file-types. Either this parameter or the
    `document` parameter must be provided.
    """

    model: Optional[str]
    """The version of the model to use for parsing."""

    output_save_url: Optional[str]
    """
    If zero data retention (ZDR) is enabled, you must enter a URL for the parsed
    output to be saved to. When ZDR is enabled, the parsed content will not be in
    the API response.
    """

    split: Optional[Literal["page"]]
    """
    If you want to split documents into smaller sections, include the split
    parameter. Set the parameter to page to split documents at the page level. The
    splits object in the API output will contain a set of data for each page.
    """
