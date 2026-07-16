from __future__ import annotations

from typing import Dict, Optional

from ..._models import BaseModel

__all__ = ["V2ExtractBilling", "V2ExtractMetadata", "V2ExtractResult"]


class V2ExtractBilling(BaseModel):
    """Billing summary: the service tier the request ran in and the credits charged."""

    service_tier: Optional[str] = None
    total_credits: Optional[float] = None
    # Characters (code points) in the input markdown as submitted -- the input
    # basis of the credit charge.
    input_markdown_chars: Optional[int] = None
    # Characters in the serialized extraction output -- the output basis of the
    # credit charge.
    output_extraction_chars: Optional[int] = None


class V2ExtractMetadata(BaseModel):
    job_id: str
    # Deprecated: renamed to `model_version` upstream; retained for backward
    # compatibility and populated only by older gateway responses.
    version: Optional[str] = None
    # Resolved model version (the current name for `version`).
    model_version: Optional[str] = None
    duration_ms: int
    doc_id: Optional[str] = None
    credit_usage: float = 0.0
    billing: Optional[V2ExtractBilling] = None
    # Units of every `range` offset in the response (always "unicode_codepoints").
    range_units: Optional[str] = None
    # URL of the OpenAPI spec covering this API.
    openapi_spec: Optional[str] = None


class V2ExtractResult(BaseModel):
    extraction: Dict[str, object]
    extraction_metadata: Dict[str, object]
    markdown: str
    metadata: V2ExtractMetadata
    # Present when the extraction output was delivered out-of-band (e.g. to a
    # ZDR save URL) instead of inline.
    output_ref: Optional[str] = None
