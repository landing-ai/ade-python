from ..._models import BaseModel

__all__ = ["ParseGroundingBox"]


class ParseGroundingBox(BaseModel):
    bottom: float

    left: float

    right: float

    top: float
