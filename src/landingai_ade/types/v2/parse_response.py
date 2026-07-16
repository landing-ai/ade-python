from __future__ import annotations

from typing import List, Optional

from ..._models import BaseModel

__all__ = [
    "V2ParseBilling",
    "V2ParseMetadata",
    "V2ParseBox",
    "V2ParseRange",
    "V2ParseNodeGrounding",
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
    # Deprecated: renamed to `output_markdown_chars` upstream; retained for
    # backward compatibility and populated only by older gateway responses.
    markdown_chars: Optional[int] = None
    # Number of Unicode code points in the returned `markdown` string.
    output_markdown_chars: Optional[int] = None
    # Units of every `range` offset in the response (always "unicode_codepoints").
    range_units: Optional[str] = None
    # URL of the OpenAPI spec covering this API.
    openapi_spec: Optional[str] = None
    failed_pages: Optional[List[int]] = None
    duration_ms: Optional[int] = None
    billing: Optional[V2ParseBilling] = None


# --- per-node spatial grounding ------------------------------------------------
#
# Every node below the `structure` root carries its spatial data inline in a
# `grounding` object (`{page, range, box}`, normalized page coordinates). The
# same shape is reused for element nodes, page nodes, and each `atomic_grounding`
# segment.


class V2ParseBox(BaseModel):
    """Axis-aligned bounding box in normalized page coordinates (`[0, 1]` fractions
    of the page width/height)."""

    xmin: Optional[float] = None
    ymin: Optional[float] = None
    xmax: Optional[float] = None
    ymax: Optional[float] = None


class V2ParseRange(BaseModel):
    """A `[start, end)` slice of the top-level `markdown` string (code-point offsets)."""

    start: Optional[int] = None
    end: Optional[int] = None


class V2ParseNodeGrounding(BaseModel):
    """Where a node lives: its 1-indexed `page`, its `range` in `markdown`, and its `box`."""

    page: Optional[int] = None
    range: Optional[V2ParseRange] = None
    box: Optional[V2ParseBox] = None


# --- `structure`: the logical document tree ------------------------------------
#
# `type` and `status` are typed as permissive `str` (not `Literal`) so a novel
# element/status value from the gateway never fails deserialization; unknown keys
# are retained via `BaseModel`'s `extra="allow"`.


class V2ParseElement(BaseModel):
    """A document element (text, table, table_cell, figure, ...). Keys off `type`;
    optional fields are only present for the relevant element types."""

    type: str
    id: str
    # Deprecated: replaced by `grounding.range` upstream; populated only by older
    # gateway responses.
    span: Optional[List[int]] = None
    # The element's spatial data (`{page, range, box}`), inline on the node.
    grounding: Optional[V2ParseNodeGrounding] = None
    # Fine-grained grounding segments (visual lines today). Present on leaf
    # elements only; omitted entirely when `options.atomic_grounding` is false.
    atomic_grounding: Optional[List[V2ParseNodeGrounding]] = None
    # The element's slice of the top-level `markdown`; only when
    # `options.inline_markdown` is true.
    markdown: Optional[str] = None
    # The cells of a `table` element (each a `table_cell`); only set for tables.
    children: Optional[List["V2ParseElement"]] = None
    # Table-cell geometry; only set on `table_cell` elements.
    row: Optional[int] = None
    col: Optional[int] = None
    colspan: Optional[int] = None
    rowspan: Optional[int] = None


class V2ParsePage(BaseModel):
    type: str = "page"
    # Deprecated: page number is now carried on `grounding.page` (1-indexed);
    # populated only by older responses.
    page: Optional[int] = None
    # Deprecated: replaced by `grounding.range` upstream.
    span: Optional[List[int]] = None
    # Deprecated: pixel dimensions/DPI were dropped in favor of normalized
    # coordinates; retained for backward compatibility.
    width: Optional[int] = None
    height: Optional[int] = None
    dpi: Optional[int] = None
    # The page's spatial data (`{page, range, box}`); `box` is the full page.
    grounding: Optional[V2ParseNodeGrounding] = None
    # This page's slice of the top-level `markdown`; only when
    # `options.inline_markdown` is true.
    markdown: Optional[str] = None
    status: str = "ok"
    reason: Optional[str] = None
    children: List[V2ParseElement] = []


class V2ParseStructure(BaseModel):
    """Root of the `structure` tree (`document -> page -> element`)."""

    type: str = "document"
    # The full document markdown; only when `options.inline_markdown` is true.
    markdown: Optional[str] = None
    children: List[V2ParsePage] = []


# --- `grounding`: legacy tree mirroring `structure` (older responses only) ------
#
# Deprecated: newer gateway responses carry per-node grounding inline on
# `structure` (see `V2ParseNodeGrounding`) and omit the top-level `grounding`
# tree. These types are retained for backward compatibility.


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
    parts: List[V2ParseGroundingEntry] = []
    # The cells of a `table` element; only set for tables.
    children: Optional[List["V2ParseGroundingElement"]] = None


class V2ParseGroundingPage(BaseModel):
    type: str = "page"
    page: int
    span: List[int]
    children: List[V2ParseGroundingElement] = []


class V2ParseGrounding(BaseModel):
    """Root of the (legacy) `grounding` tree, mirroring `structure` with spatial data."""

    type: str = "document"
    children: List[V2ParseGroundingPage] = []


class V2ParseResponse(BaseModel):
    """V2 parse result. `structure` is a typed tree mirroring the published gateway
    schema, with per-node spatial `grounding` inline; unknown element types and
    extra keys are retained. `grounding` is the legacy top-level tree, present only
    on older gateway responses."""

    markdown: Optional[str] = None
    structure: Optional[V2ParseStructure] = None
    grounding: Optional[V2ParseGrounding] = None
    metadata: Optional[V2ParseMetadata] = None
