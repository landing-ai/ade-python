# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from .._models import BaseModel

__all__ = ["ParseJobListResponse", "Job"]


class Job(BaseModel):
    """Summary of a job for listing."""

    job_id: str

    progress: float
    """
    Job completion progress as a decimal from 0 to 1, where 0 is not started, 1 is
    finished, and values between 0 and 1 indicate work in progress.
    """

    received_at: int

    status: str

    failure_reason: Optional[str] = None


class ParseJobListResponse(BaseModel):
    """Response for listing jobs."""

    jobs: List[Job]

    has_more: Optional[bool] = None

    org_id: Optional[str] = None
