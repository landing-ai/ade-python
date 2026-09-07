from __future__ import annotations

from typing import List, Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["V2Classification", "V2ClassifyMetadata", "V2ClassifyResponse"]


class V2Classification(BaseModel):
    """A single page-level classification result."""

    class_: str = FieldInfo(alias="class")
    """Predicted class label, or 'unknown'."""

    page: int
    """Page number, zero-indexed (the first page is 0)."""

    reason: str
    """Why the page was classified this way."""

    suggested_class: Optional[str] = None
    """A class the model proposes when the prediction is 'unknown'."""


class V2ClassifyMetadata(BaseModel):
    """Response metadata for a V2 classify call."""

    # Required and non-null per the spec.
    page_count: int
    duration_ms: int
    # URL of the OpenAPI spec covering this API. Required and non-null per the spec.
    openapi_spec: str
    credit_usage: Optional[float] = None
    filename: Optional[str] = None
    # Gateway job id (workflow id). Matches the billing row id in vision-agent.
    job_id: Optional[str] = None
    org_id: Optional[str] = None
    # Resolved classify pipeline version that produced this response.
    version: Optional[str] = None


class V2ClassifyResponse(BaseModel):
    """Response model for the V2 classify endpoint."""

    classification: List[V2Classification]
    """One classification result per page, in page order."""

    metadata: V2ClassifyMetadata
