# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, TypedDict

from .._types import FileTypes

__all__ = ["ClientParseParams"]


class ClientParseParams(TypedDict, total=False):
    custom_prompts: Optional[str]
    """Optional JSON string mapping chunk types to custom parsing prompts.

    Only the `figure` key is supported, for example '{"figure":"Describe axis labels
    in detail."}'.
    """

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

    password: Optional[str]
    """Password for encrypted document files.

    If the document is password-protected, provide the password to decrypt and
    process the document. Ignored for unencrypted documents.
    """

    split: Optional[Literal["page"]]
    """
    If you want to split documents into smaller sections, include the split
    parameter. Set the parameter to page to split documents at the page level. The
    splits object in the API output will contain a set of data for each page.
    """
