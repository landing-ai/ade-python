from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from ..._models import BaseModel

__all__ = [
    "V2ParseBilling",
    "V2ParseMetadata",
    "V2ParseElement",
    "V2ParsePage",
    "V2ParseStructure",
    "V2ParseGroundingEntry",
    "V2ParseGroundingElement",
    "V2ParseGroundingPage",
    "V2ParseGrounding",
    "V2ParseResponse",
]


class V2ParseBilling(BaseModel):
    service_tier: Optional[str] = None
    total_credits: Optional[float] = None


class V2ParseMetadata(BaseModel):
    req_id: Optional[str] = None
    job_id: Optional[str] = None
    model_version: Optional[str] = None
    page_count: Optional[int] = None
    markdown_chars: Optional[int] = None
    failed_pages: Optional[List[int]] = None
    duration_ms: Optional[int] = None
    billing: Optional[V2ParseBilling] = None


# --- `structure`: the logical document tree (no spatial data) ------------------
#
# `type` and `status` are typed as permissive `str` (not `Literal`) so a novel
# element/status value from the gateway never fails deserialization; unknown keys
# are retained via `BaseModel`'s `extra="allow"`.


class V2ParseElement(BaseModel):
    """A document element (text, table, table_cell, figure, ...). Keys off `type`;
    optional fields are only present for the relevant element types."""

    type: str
    id: str
    # Unicode code-point offsets `[start, end)` into the top-level `markdown`.
    span: List[int]
    # The cells of a `table` element (each a `table_cell`); only set for tables.
    children: Optional[List["V2ParseElement"]] = None
    # Table-cell geometry; only set on `table_cell` elements.
    row: Optional[int] = None
    col: Optional[int] = None
    colspan: Optional[int] = None
    rowspan: Optional[int] = None


class V2ParsePage(BaseModel):
    type: str = "page"
    # 0-indexed page number in the source document.
    page: int
    span: List[int]
    width: Optional[int] = None
    height: Optional[int] = None
    dpi: Optional[int] = None
    status: str = "ok"
    reason: Optional[str] = None
    children: List[V2ParseElement] = Field(default_factory=list)


class V2ParseStructure(BaseModel):
    """Root of the `structure` tree (`document -> page -> element`)."""

    type: str = "document"
    children: List[V2ParsePage] = Field(default_factory=list)


# --- `grounding`: mirrors `structure`, carrying the spatial data ---------------


class V2ParseGroundingEntry(BaseModel):
    """One fine-grained grounding segment (line-level or finer)."""

    span: List[int]
    # Bounding box `[left, top, right, bottom]` in source-page pixels.
    box: List[int]


class V2ParseGroundingElement(BaseModel):
    type: str
    id: str
    span: List[int]
    box: List[int]
    parts: List[V2ParseGroundingEntry] = Field(default_factory=list)
    # The cells of a `table` element; only set for tables.
    children: Optional[List["V2ParseGroundingElement"]] = None


class V2ParseGroundingPage(BaseModel):
    type: str = "page"
    page: int
    span: List[int]
    children: List[V2ParseGroundingElement] = Field(default_factory=list)


class V2ParseGrounding(BaseModel):
    """Root of the `grounding` tree, mirroring `structure` with spatial data."""

    type: str = "document"
    children: List[V2ParseGroundingPage] = Field(default_factory=list)


class V2ParseResponse(BaseModel):
    """V2 parse result. `structure` and `grounding` are typed trees mirroring the
    published gateway schema; unknown element types and extra keys are retained."""

    markdown: Optional[str] = None
    structure: Optional[V2ParseStructure] = None
    grounding: Optional[V2ParseGrounding] = None
    metadata: Optional[V2ParseMetadata] = None
