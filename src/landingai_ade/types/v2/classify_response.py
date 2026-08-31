from __future__ import annotations

from typing import List, Optional

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = ["V2ClassificationItem", "V2ClassifyMetadata", "V2ClassifyResponse"]


class V2ClassificationItem(BaseModel):
    """One page-level classification result."""

    # Predicted class label, or "unknown". The wire key is `class` (a Python
    # keyword), so the field is `class_` with the alias carrying the wire name.
    class_: str = FieldInfo(alias="class")
    # Page number, zero-indexed (the first page is 0).
    page: int
    # Why the page was classified this way.
    reason: str
    # A class the model proposes when the prediction is "unknown".
    suggested_class: Optional[str] = None


class V2ClassifyMetadata(BaseModel):
    """Response metadata for a v2 classify call."""

    # Number of pages classified. Required and non-null per the spec.
    page_count: int
    # End-to-end request duration in milliseconds. Required and non-null per the spec.
    duration_ms: int
    # URL of the OpenAPI spec covering this API. Required and non-null per the spec.
    openapi_spec: str
    # Credits billed for this request.
    credit_usage: Optional[float] = None
    # Name of the classified file.
    filename: Optional[str] = None
    # Gateway job id (workflow id). Matches the billing row id in vision-agent.
    job_id: Optional[str] = None
    org_id: Optional[str] = None
    # Resolved classify pipeline version that produced this response.
    version: Optional[str] = None


class V2ClassifyResponse(BaseModel):
    # One classification result per page, in page order.
    classification: List[V2ClassificationItem]
    metadata: V2ClassifyMetadata
