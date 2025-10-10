# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["ParseJobListParams"]


class ParseJobListParams(TypedDict, total=False):
    page: int
    """Page number (0-indexed)"""

    page_size: Annotated[int, PropertyInfo(alias="pageSize")]
    """Number of items per page"""

    status: Optional[Literal["cancelled", "completed", "failed", "pending", "processing"]]
    """Filter by job status."""
