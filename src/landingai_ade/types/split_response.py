# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from .._models import BaseModel

__all__ = ["SplitResponse", "Metadata", "Split"]


class Metadata(BaseModel):
    """Metadata for split classification response."""

    credit_usage: float

    duration_ms: int

    filename: str

    page_count: int

    job_id: Optional[str] = None
    """Inference history job ID"""

    org_id: Optional[str] = None
    """Organization ID"""

    version: Optional[str] = None
    """Model version used for split classification"""


class Split(BaseModel):
    """Split data for split classification endpoint."""

    classification: str

    identifier: Optional[str] = None

    markdowns: List[str]

    pages: List[int]


class SplitResponse(BaseModel):
    """Response model for split classification endpoint."""

    metadata: Metadata
    """Metadata for split classification response."""

    splits: List[Split]
