# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .._models import BaseModel

__all__ = ["ExtractJobCreateResponse"]


class ExtractJobCreateResponse(BaseModel):
    job_id: str
    """The unique identifier for the extract job."""

    message: Optional[str] = None
    """Optional message about the job creation."""