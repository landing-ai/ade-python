# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .._models import BaseModel

__all__ = ["ExtractResponse", "Metadata"]


class Metadata(BaseModel):
    credit_usage: float

    duration_ms: int

    filename: str

    job_id: str

    org_id: Optional[str] = None

    version: Optional[str] = None


class ExtractResponse(BaseModel):
    extraction: object
    """The extracted key-value pairs."""

    extraction_metadata: object
    """The extracted key-value pairs and the chunk_reference for each one."""

    metadata: Metadata
    """The metadata for the extraction process."""
