# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .._models import BaseModel

__all__ = ["ExtractResponse", "Metadata"]


class Metadata(BaseModel):
    """The metadata for the extraction process."""

    credit_usage: float

    duration_ms: int

    filename: str

    job_id: str

    org_id: Optional[str] = None

    version: Optional[str] = None

    fallback_model_version: Optional[str] = None
    """
    The extract model that was actually used to extract the data when the initial
    extraction attempt failed with the requested version.
    """

    schema_violation_error: Optional[str] = None
    """
    A detailed error message shows why the extracted data does not fully conform to
    the input schema. Null means the extraction result is consistent with the input
    schema.
    """


class ExtractResponse(BaseModel):
    extraction: object
    """The extracted key-value pairs."""

    extraction_metadata: object
    """The extracted key-value pairs and the chunk_reference for each one."""

    metadata: Metadata
    """The metadata for the extraction process."""
