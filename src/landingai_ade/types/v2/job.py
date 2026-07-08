from __future__ import annotations

from enum import Enum
from typing import Dict, Optional
from datetime import datetime

from pydantic import Field

from ..._models import BaseModel

__all__ = ["JobStatus", "JobError", "Job"]


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobError(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None


class Job(BaseModel):
    """One normalized job shape across parse and extract (envelopes diverge upstream)."""

    job_id: str
    status: JobStatus
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = None
    # Populated on completion: V2ParseResponse for parse jobs, V2ExtractResult for extract jobs.
    result: Optional[object] = None
    error: Optional[JobError] = None
    # Full original envelope for fields not surfaced above (org_id, output_url, version, ...).
    raw: Dict[str, object] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
