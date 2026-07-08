from .._models import BaseModel

__all__ = ["ParseJobCreateResponse"]


class ParseJobCreateResponse(BaseModel):
    job_id: str
