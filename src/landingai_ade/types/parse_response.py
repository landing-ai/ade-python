# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Optional
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from .._models import BaseModel

__all__ = [
    "ParseResponse",
    "Chunk",
    "ChunkGrounding",
    "ChunkGroundingBox",
    "Metadata",
    "Split",
    "Grounding",
    "GroundingBox",
]


class ChunkGroundingBox(BaseModel):
    bottom: float

    left: float

    right: float

    top: float


class ChunkGrounding(BaseModel):
    box: ChunkGroundingBox

    page: int


class Chunk(BaseModel):
    id: str

    grounding: ChunkGrounding

    markdown: str

    type: str


class Metadata(BaseModel):
    credit_usage: float

    duration_ms: int

    filename: str

    job_id: str

    org_id: Optional[str] = None

    page_count: int

    version: Optional[str] = None


class Split(BaseModel):
    chunks: List[str]

    class_: str = FieldInfo(alias="class")

    identifier: str

    markdown: str

    pages: List[int]


class GroundingBox(BaseModel):
    bottom: float

    left: float

    right: float

    top: float


class Grounding(BaseModel):
    box: GroundingBox

    page: int

    type: Literal[
        "chunkLogo",
        "chunkCard",
        "chunkAttestation",
        "chunkScanCode",
        "chunkForm",
        "chunkTable",
        "chunkFigure",
        "chunkText",
        "chunkMarginalia",
        "chunkTitle",
        "chunkPageHeader",
        "chunkPageFooter",
        "chunkPageNumber",
        "chunkKeyValue",
        "table",
        "tableCell",
    ]


class ParseResponse(BaseModel):
    chunks: List[Chunk]

    markdown: str

    metadata: Metadata

    splits: List[Split]

    grounding: Optional[Dict[str, Grounding]] = None
