# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Union, Optional
from typing_extensions import Literal, TypeAlias

from pydantic import Field as FieldInfo

from .._models import BaseModel
from .shared.parse_metadata import ParseMetadata
from .shared.parse_grounding_box import ParseGroundingBox

__all__ = [
    "ParseResponse",
    "Chunk",
    "ChunkGrounding",
    "Split",
    "Grounding",
    "GroundingParseResponseGrounding",
    "GroundingParseResponseTableCellGrounding",
    "GroundingParseResponseTableCellGroundingPosition",
]


class ChunkGrounding(BaseModel):
    box: ParseGroundingBox

    page: int


class Chunk(BaseModel):
    id: str

    grounding: ChunkGrounding

    markdown: str

    type: str


class Split(BaseModel):
    chunks: List[str]

    class_: str = FieldInfo(alias="class")

    identifier: str

    markdown: str

    pages: List[int]


class GroundingParseResponseGrounding(BaseModel):
    box: ParseGroundingBox

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


class GroundingParseResponseTableCellGroundingPosition(BaseModel):
    chunk_id: str

    col: int

    colspan: int

    row: int

    rowspan: int


class GroundingParseResponseTableCellGrounding(BaseModel):
    box: ParseGroundingBox

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

    position: Optional[GroundingParseResponseTableCellGroundingPosition] = None


Grounding: TypeAlias = Union[GroundingParseResponseGrounding, GroundingParseResponseTableCellGrounding]


class ParseResponse(BaseModel):
    chunks: List[Chunk]

    markdown: str

    metadata: ParseMetadata

    splits: List[Split]

    grounding: Optional[Dict[str, Grounding]] = None
