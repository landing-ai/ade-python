from typing import List, Optional

from .._models import BaseModel

__all__ = ["ExtractJobListResponse", "Job"]


class Job(BaseModel):
    """Summary of a job for listing."""

    job_id: str

    progress: float
    """Job completion as a decimal from 0 (not started) to 1 (complete)."""

    received_at: int

    status: str

    failure_reason: Optional[str] = None


class ExtractJobListResponse(BaseModel):
    """Response for listing jobs."""

    jobs: List[Job]

    has_more: Optional[bool] = None

    org_id: Optional[str] = None
