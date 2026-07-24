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
    """A structured warning from the schema-generation process (`{code, msg}`)."""

    # The type of warning, used to translate to a status code downstream
    # (e.g. `nonconformant_schema`). Required and non-null per the spec.
    code: str
    # Human-readable description of the warning with more details. Required and
    # non-null per the spec.
    msg: str


class V2BuildSchemaBilling(BaseModel):
    """Billing summary: the service tier the request ran in and the credits charged."""

    service_tier: Optional[str] = None
    total_credits: Optional[float] = None


class V2BuildSchemaMetadata(BaseModel):
    """Response metadata for a v2 build-schema call."""

    # Gateway job id (workflow id). Matches the billing row id in vision-agent.
    # Spec default is the empty string, not null.
    job_id: Optional[str] = ""
    # End-to-end request duration in milliseconds. Spec default is 0, not null.
    duration_ms: Optional[int] = 0
    # Name of the first source document. Retained for v1 compatibility but NOT
    # populated in this version -- always None.
    filename: Optional[str] = None
    # Organization ID.
    org_id: Optional[str] = None
    # Model version used for generation. build-schema is version-free, so this is
    # always None; retained for v1 response-shape compatibility.
    version: Optional[str] = None
    # URL of the OpenAPI spec covering this API. Required and non-null per the spec.
    openapi_spec: str
    billing: Optional[V2BuildSchemaBilling] = None
    # Structured warnings from the schema-generation process, each a `{code, msg}`.
    warnings: Optional[List[V2BuildSchemaWarning]] = None


class V2BuildSchemaResponse(BaseModel):
    # The generated JSON Schema serialized as a STRING (VTRA parity -- the field
    # is a string, not an object).
    extraction_schema: str
    metadata: V2BuildSchemaMetadata
