# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ExtractResponse", "Metadata", "MetadataWarning"]


class MetadataWarning(BaseModel):
    code: Literal["nonconformant_schema", "nonconformant_output"]
    """The type of warning, used to translate to a status code downstream"""

    msg: str
    """Human-readable description of the warning with more details"""


class Metadata(BaseModel):
    """The metadata for the extraction process."""

    credit_usage: float

    duration_ms: int

    filename: str

    job_id: str

    org_id: Optional[str] = None

    version: Optional[str] = None

    fallback_model_version: Optional[str] = None
    """
    The extract model that was actually used to extract the data when the initial
    extraction attempt failed with the requested version.
    """

    schema_violation_error: Optional[str] = None
    """
    A detailed error message shows why the extracted data does not fully conform to
    the input schema. Null means the extraction result is consistent with the input
    schema.
    """

    warnings: Optional[List[MetadataWarning]] = None
    """Structured warnings from the extraction process.

    Each warning is an instance of ExtractWarning with 'code' (e.g.
    'nonconformant_schema') and 'msg' (human-readable description). Present only for
    extract versions from extract-20260314 and above that support structured
    warnings.
    """


class ExtractResponse(BaseModel):
    extraction: object
    """The extracted key-value pairs."""

    extraction_metadata: object
    """The extracted key-value pairs and the chunk_reference for each one."""

    metadata: Metadata
    """The metadata for the extraction process."""
