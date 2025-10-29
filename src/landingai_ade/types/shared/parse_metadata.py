# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from ..._models import BaseModel

__all__ = ["ParseMetadata"]


class ParseMetadata(BaseModel):
    credit_usage: float

    duration_ms: int

    filename: str

    job_id: str

    org_id: Optional[str] = None

    page_count: int

    version: Optional[str] = None

    failed_pages: Optional[List[int]] = None
