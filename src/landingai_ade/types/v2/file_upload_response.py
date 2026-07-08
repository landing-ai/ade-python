from __future__ import annotations

from typing import Optional

from ..._models import BaseModel

__all__ = ["V2FileUploadResponse"]


class V2FileUploadResponse(BaseModel):
    """`POST /v1/files` returns an open string map; `file_ref` is the key we consume."""

    file_ref: Optional[str] = None
