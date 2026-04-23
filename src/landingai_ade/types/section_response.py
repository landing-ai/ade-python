# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from .._models import BaseModel

__all__ = ["SectionResponse", "Metadata", "TableOfContent"]


class Metadata(BaseModel):
    """Public metadata for section response."""

    credit_usage: float

    duration_ms: int

    filename: str

    job_id: Optional[str] = None

    org_id: Optional[str] = None

    version: Optional[str] = None


class TableOfContent(BaseModel):
    """A single entry in the flat table of contents."""

    level: int

    section_number: str

    start_reference: str

    title: str


class SectionResponse(BaseModel):
    """Response model for section endpoint."""

    metadata: Metadata
    """Public metadata for section response."""

    table_of_contents: List[TableOfContent]

    table_of_contents_md: str
