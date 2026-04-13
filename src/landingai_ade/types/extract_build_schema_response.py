# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ExtractBuildSchemaResponse", "Metadata", "MetadataWarning"]


class MetadataWarning(BaseModel):
    code: Literal["nonconformant_schema", "nonconformant_output"]
    """The type of warning, used to translate to a status code downstream"""

    msg: str
    """Human-readable description of the warning with more details"""


class Metadata(BaseModel):
    """The metadata for the schema generation process."""

    credit_usage: Optional[float] = None

    duration_ms: Optional[int] = None

    filename: Optional[str] = None

    job_id: Optional[str] = None

    org_id: Optional[str] = None

    version: Optional[str] = None

    warnings: Optional[List[MetadataWarning]] = None
    """Structured warnings from the extraction process.

    Each warning is an instance of ExtractWarning with 'code' (e.g.
    'nonconformant_schema') and 'msg' (human-readable description). Present only for
    extract versions from extract-20260314 and above that support structured
    warnings.
    """


class ExtractBuildSchemaResponse(BaseModel):
    extraction_schema: str
    """The generated JSON schema as a string."""

    metadata: Metadata
    """The metadata for the schema generation process."""
