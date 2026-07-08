from __future__ import annotations

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["V2ParseBilling", "V2ParseMetadata", "V2ParseResponse"]


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


class V2ParseResponse(BaseModel):
    """V2 parse result. The gateway spec types this loosely; fields are permissive
    and extra keys are retained. Re-verify against the typed schema when the gateway
    publishes one."""

    markdown: Optional[str] = None
    structure: Optional[object] = None
    grounding: Optional[object] = None
    metadata: Optional[V2ParseMetadata] = None
