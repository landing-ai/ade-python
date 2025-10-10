# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from ..._models import BaseModel

__all__ = ["ParseGroundingBox"]


class ParseGroundingBox(BaseModel):
    bottom: float

    left: float

    right: float

    top: float
