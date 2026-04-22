# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = ["ClassifyResponse", "Classification", "Metadata"]


class Classification(BaseModel):
    """A single page-level classification result."""

    class_: str = FieldInfo(alias="class")
    """Predicted class label or 'unknown'."""

    page: int
    """Page number (0-based)."""

    reason: Optional[str] = None
    """Reason for the classification (for debugging)."""

    suggested_class: Optional[str] = None
    """Proposed class when the prediction is 'unknown'."""


class Metadata(BaseModel):
    """Metadata for the classify response."""

    credit_usage: float

    duration_ms: int

    filename: str

    page_count: int

    job_id: Optional[str] = None

    org_id: Optional[str] = None

    version: Optional[str] = None


class ClassifyResponse(BaseModel):
    """Response model for the classify endpoint."""

    classification: List[Classification]

    metadata: Metadata
    """Metadata for the classify response."""
