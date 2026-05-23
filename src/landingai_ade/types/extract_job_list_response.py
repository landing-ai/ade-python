# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List

from .._models import BaseModel
from .extract_job_get_response import ExtractJobGetResponse

__all__ = ["ExtractJobListResponse"]


class ExtractJobListResponse(BaseModel):
    has_more: bool
    """Whether there are more results available."""

    list: List[ExtractJobGetResponse]
    """List of extract jobs."""