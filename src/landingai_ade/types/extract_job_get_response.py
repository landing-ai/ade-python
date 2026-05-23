# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Any, Dict, List, Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["ExtractJobGetResponse"]


class ExtractJobGetResponse(BaseModel):
    credit_usage: float
    """The number of credits used for this job."""

    error_pages: int
    """The number of pages that failed to process."""

    id: str
    """The unique identifier for the extract job."""

    processed_pages: int
    """The number of pages successfully processed.

    For extract jobs, this represents the number of characters processed."""

    received_at: str
    """The timestamp when the job was received."""

    skipped_pages: int
    """The number of pages skipped during processing."""

    status: Literal["pending", "processing", "completed", "failed", "cancelled"]
    """The current status of the job."""

    total_pages: int
    """The total number of pages in the document.

    For extract jobs, this represents the total number of characters."""

    version: str
    """The model version used for extraction."""

    document_url: Optional[str] = None
    """The URL of the document (if provided via URL)."""

    extracted_data: Optional[Dict[str, Any]] = None
    """The extracted data according to the provided schema.

    This field is only present when the job is completed successfully and
    zero data retention is not enabled."""

    failed_page_numbers: Optional[List[int]] = None
    """The list of page numbers that failed to process."""

    failure_reason: Optional[str] = None
    """The reason for job failure (if applicable)."""

    filename: Optional[str] = None
    """The name of the processed file."""

    file_size: Optional[int] = None
    """The size of the processed file in bytes."""

    finished_at: Optional[str] = None
    """The timestamp when the job finished processing."""

    inference_history_id: Optional[str] = None
    """The ID for tracking this inference in history."""

    org_id: Optional[str] = None
    """The organization ID associated with this job."""

    output_save_url: Optional[str] = None
    """The URL where the output was saved (for ZDR mode)."""

    output_url: Optional[str] = None
    """The URL to retrieve the extraction results."""

    processing_attempts: Optional[int] = None
    """The number of processing attempts made for this job."""

    schema_violation_error: Optional[Dict[str, Any]] = None
    """Details about any schema validation errors encountered."""