from __future__ import annotations

from typing import Dict, Optional

from ..._models import BaseModel

__all__ = ["V2ExtractMetadata", "V2ExtractResult"]


class V2ExtractMetadata(BaseModel):
    job_id: str
    version: str
    duration_ms: int
    doc_id: Optional[str] = None
    credit_usage: float = 0.0


class V2ExtractResult(BaseModel):
    extraction: Dict[str, object]
    extraction_metadata: Dict[str, object]
    markdown: str
    metadata: V2ExtractMetadata
