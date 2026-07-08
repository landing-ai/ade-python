from typing import List, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = [
    "ExtractJobGetResponse",
    "Data",
    "DataMetadata",
    "DataMetadataWarning",
    "Metadata",
    "MetadataWarning",
]


class DataMetadataWarning(BaseModel):
    code: Literal["nonconformant_schema", "nonconformant_output"]
    """The type of warning, used to translate to a status code downstream"""

    msg: str
    """Human-readable description of the warning with more details"""


class DataMetadata(BaseModel):
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

    warnings: Optional[List[DataMetadataWarning]] = None
    """Structured warnings from the extraction process.

    Each warning is an instance of ExtractWarning with 'code' (e.g.
    'nonconformant_schema') and 'msg' (human-readable description). Present only for
    extract versions from extract-20260314 and above that support structured
    warnings.
    """


class Data(BaseModel):
    extraction: object
    """The extracted key-value pairs."""

    extraction_metadata: object
    """The extracted key-value pairs and the chunk_reference for each one."""

    metadata: DataMetadata
    """The metadata for the extraction process."""


class MetadataWarning(BaseModel):
    code: Literal["nonconformant_schema", "nonconformant_output"]
    """The type of warning, used to translate to a status code downstream"""

    msg: str
    """Human-readable description of the warning with more details"""


class Metadata(BaseModel):
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


class ExtractJobGetResponse(BaseModel):
    """The status of an extract job, plus the results once it completes."""

    job_id: str
    """A unique identifier for this extract job."""

    progress: float
    """Job completion. Either 0.0 (not yet complete) or 1.0 (complete)."""

    received_at: int
    """Unix timestamp (in seconds) for when the job was received."""

    status: str
    """
    The current state of the job: `pending`, `processing`, `completed`, `failed`, or
    `cancelled`.
    """

    data: Optional[Data] = None
    """
    The extraction results, returned here when the job is complete and you did not
    set an `output_save_url`. Large results are returned through `output_url`
    instead.
    """

    failure_reason: Optional[str] = None
    """If the job failed, a message describing what went wrong."""

    metadata: Optional[Metadata] = None
    """
    Information about the extraction, such as the model version, duration, credit
    usage, and any schema warnings.
    """

    org_id: Optional[str] = None
    """Organization ID."""

    output_url: Optional[str] = None
    """A URL to download the extraction results.

    Provided when the job is complete and either you set an `output_save_url` or the
    result is larger than 1 MB. URLs for large results are temporary and expire one
    hour after you request the job.
    """

    version: Optional[str] = None
    """The exact model snapshot used for the extraction."""
