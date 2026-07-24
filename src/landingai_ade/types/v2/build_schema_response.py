from __future__ import annotations

from typing import List, Optional

from ..._models import BaseModel

__all__ = [
    "V2BuildSchemaWarning",
    "V2BuildSchemaBilling",
    "V2BuildSchemaMetadata",
    "V2BuildSchemaResponse",
]


class V2BuildSchemaWarning(BaseModel):
    """A structured warning from the schema-generation process."""

    # The type of warning (e.g. `nonconformant_schema`), used to translate to a
    # status code downstream.
    code: Optional[str] = None
    # Human-readable description of the warning with more details.
    msg: Optional[str] = None


class V2BuildSchemaBilling(BaseModel):
    """Billing summary: the service tier the request ran in and the credits charged."""

    service_tier: Optional[str] = None
    total_credits: Optional[float] = None


class V2BuildSchemaMetadata(BaseModel):
    """Response metadata for a v2 build-schema call."""

    # Gateway job id (workflow id). Matches the billing row id in vision-agent.
    job_id: Optional[str] = None
    duration_ms: Optional[int] = None
    # Name of the first source document. Retained for v1 compatibility but not
    # populated in this version -- always None.
    filename: Optional[str] = None
    org_id: Optional[str] = None
    # Model version used for generation. build-schema is version-free, so this is
    # always None; retained for v1 response-shape compatibility.
    version: Optional[str] = None
    # URL of the OpenAPI spec covering this API.
    openapi_spec: Optional[str] = None
    # Structured warnings from the schema-generation process.
    warnings: Optional[List[V2BuildSchemaWarning]] = None
    billing: Optional[V2BuildSchemaBilling] = None


class V2BuildSchemaResponse(BaseModel):
    # The generated JSON schema serialized as a string (VTRA parity -- the field
    # is a string, not an object).
    extraction_schema: str
    metadata: V2BuildSchemaMetadata
