# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, TypedDict

__all__ = ["ExtractJobListParams"]


class ExtractJobListParams(TypedDict, total=False):
    ending_before: Optional[str]
    """A cursor to retrieve results before a specific job ID."""

    limit: int
    """Number of results to return (default 10, max 100)."""

    starting_after: Optional[str]
    """A cursor to retrieve results after a specific job ID."""

    status: Optional[Literal["pending", "processing", "completed", "failed", "cancelled"]]
    """Filter by job status."""