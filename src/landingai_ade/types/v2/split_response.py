from __future__ import annotations

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["V2Split", "V2SplitMetadata", "V2SplitResponse"]


class V2Split(BaseModel):
    """One split segment: consecutive pages of one classification.

    An identifier change starts a new segment.
    """

    # The split classification name this segment was assigned.
    classification: str
    # The Markdown content of each page in this segment, in order.
    markdowns: List[str]
    # 0-indexed page numbers belonging to this segment, in order.
    pages: List[int]
    # The identifier value extracted for this segment, when the matching split
    # classification requested one. Null otherwise (the key is always present per
    # the spec, but may be null).
    identifier: Optional[str] = None


class V2SplitMetadata(BaseModel):
    """Information about a split request."""

    # Credits consumed by this request. Required and non-null per the spec.
    credit_usage: float
    # Total processing time in milliseconds. Required and non-null per the spec.
    duration_ms: int
    # Display name of the split document. Required and non-null per the spec.
    filename: str
    # The split job identifier -- server-minted and unique per request. Required
    # and non-null per the spec.
    job_id: str
    # Total number of pages in the input Markdown. Required and non-null per the spec.
    page_count: int
    # The exact split model snapshot that processed the document, e.g.
    # `split-20251105`. Required and non-null per the spec.
    version: str
    org_id: Optional[str] = None


class V2SplitResponse(BaseModel):
    # The split segments, in page order. Consecutive pages with the same
    # classification merge into one segment; an identifier change starts a new one.
    splits: List[V2Split]
    metadata: V2SplitMetadata
