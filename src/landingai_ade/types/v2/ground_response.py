from __future__ import annotations

from typing import Dict, Optional

from ..._models import BaseModel

__all__ = ["V2GroundBilling", "V2GroundMetadata", "V2GroundResult"]


class V2GroundBilling(BaseModel):
    """Billing summary: the service tier the request ran in and the credits charged."""

    service_tier: Optional[str] = None
    total_credits: Optional[float] = None


class V2GroundMetadata(BaseModel):
    """Response metadata for a v2 ground call."""

    job_id: str
    duration_ms: int
    # URL of the OpenAPI spec covering this API.
    openapi_spec: Optional[str] = None
    billing: Optional[V2GroundBilling] = None


class V2GroundResult(BaseModel):
    # A tree mirroring the input `extraction_metadata`: each `{value, ranges}`
    # leaf is replaced by the list of `structure` blocks its ranges overlap. Not
    # a flat map -- a nested field like `issuer.name` resolves to
    # `grounding["issuer"]["name"]`.
    grounding: Dict[str, object]
    metadata: V2GroundMetadata
