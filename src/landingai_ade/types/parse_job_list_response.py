# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from .._models import BaseModel

__all__ = ["ParseJobListResponse", "Job"]


class Job(BaseModel):
    """Summary of a job for listing."""

    job_id: str

    progress: float
    """Job completion as a decimal from 0 (not started) to 1 (complete)."""

    received_at: int

    status: str

    created_at: Optional[int] = None
    """Unix timestamp (seconds) for when the job was created.

    Mirrors received_at; exposed so clients have an explicit creation time.
    """

    failure_reason: Optional[str] = None


class ParseJobListResponse(BaseModel):
    """Response for listing jobs."""

    jobs: List[Job]

    has_more: Optional[bool] = None

    org_id: Optional[str] = None
