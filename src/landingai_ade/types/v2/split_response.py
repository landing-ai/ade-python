from __future__ import annotations

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["V2Split", "V2SplitMetadata", "V2SplitResponse"]


class V2Split(BaseModel):
    """One split segment: consecutive pages of one classification.

    An identifier change starts a new segment. Every field is required per the
    spec; `identifier` is required-but-nullable (null when the matching split
    classification requested no identifier).
    """

    classification: str
    """The split classification name this segment was assigned."""

    identifier: Optional[str]
    """The identifier value extracted for this segment, or null."""

    markdowns: List[str]
    """The Markdown content of each page in this segment, in order."""

    pages: List[int]
    """0-indexed page numbers belonging to this segment, in order."""


class V2SplitMetadata(BaseModel):
    """Information about a split request. Every field is required per the spec."""

    filename: str
    org_id: Optional[str]
    page_count: int
    duration_ms: int
    credit_usage: float
    # The split job identifier -- server-minted and unique per request.
    job_id: str
    # The exact split model snapshot that processed the document, e.g. `split-20251105`.
    version: str


class V2SplitResponse(BaseModel):
    """Response model for the V2 split endpoint."""

    splits: List[V2Split]
    """The split segments, in page order."""

    metadata: V2SplitMetadata
