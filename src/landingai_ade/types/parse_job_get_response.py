# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Union, Optional
from typing_extensions import Literal, TypeAlias

from pydantic import Field as FieldInfo

from .._models import BaseModel
from .shared.parse_metadata import ParseMetadata
from .shared.parse_grounding_box import ParseGroundingBox

__all__ = [
    "ParseJobGetResponse",
    "Data",
    "DataParseResponse",
    "DataParseResponseChunk",
    "DataParseResponseChunkGrounding",
    "DataParseResponseSplit",
    "DataParseResponseGrounding",
    "DataParseResponseGroundingParseResponseGrounding",
    "DataParseResponseGroundingParseResponseTableCellGrounding",
    "DataParseResponseGroundingParseResponseTableCellGroundingPosition",
    "DataSpreadsheetParseResponse",
    "DataSpreadsheetParseResponseChunk",
    "DataSpreadsheetParseResponseChunkGrounding",
    "DataSpreadsheetParseResponseMetadata",
    "DataSpreadsheetParseResponseSplit",
]


class DataParseResponseChunkGrounding(BaseModel):
    box: ParseGroundingBox

    page: int


class DataParseResponseChunk(BaseModel):
    id: str

    grounding: DataParseResponseChunkGrounding

    markdown: str

    type: str


class DataParseResponseSplit(BaseModel):
    chunks: List[str]

    class_: str = FieldInfo(alias="class")

    identifier: str

    markdown: str

    pages: List[int]


class DataParseResponseGroundingParseResponseGrounding(BaseModel):
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


class DataParseResponseGroundingParseResponseTableCellGroundingPosition(BaseModel):
    chunk_id: str

    col: int

    colspan: int

    row: int

    rowspan: int


class DataParseResponseGroundingParseResponseTableCellGrounding(BaseModel):
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

    position: Optional[DataParseResponseGroundingParseResponseTableCellGroundingPosition] = None


DataParseResponseGrounding: TypeAlias = Union[
    DataParseResponseGroundingParseResponseGrounding, DataParseResponseGroundingParseResponseTableCellGrounding
]


class DataParseResponse(BaseModel):
    chunks: List[DataParseResponseChunk]

    markdown: str

    metadata: ParseMetadata

    splits: List[DataParseResponseSplit]

    grounding: Optional[Dict[str, DataParseResponseGrounding]] = None


class DataSpreadsheetParseResponseChunkGrounding(BaseModel):
    """
    Visual grounding coordinates from /parse API (only for chunks derived from embedded images)
    """

    box: ParseGroundingBox

    page: int


class DataSpreadsheetParseResponseChunk(BaseModel):
    """Chunk from spreadsheet parsing.

    Can represent:
    - Table chunks from spreadsheet cells
    - Parsed content chunks from embedded images (text, table, figure, etc.)
    """

    id: str
    """
    Chunk ID - format: '{sheet_name}-{cell_range}' for tables,
    '{sheet_name}-image-{index}-{anchor_cell}-chunk-{i}-{type}' for parsed image
    chunks
    """

    markdown: str
    """
    Chunk content as HTML table with anchor tag (for tables) or parsed markdown
    content (for chunks from images)
    """

    type: str
    """
    Chunk type: 'table' for spreadsheet tables, or types from /parse (text, table,
    figure, form, etc.) for chunks derived from embedded images
    """

    grounding: Optional[DataSpreadsheetParseResponseChunkGrounding] = None
    """
    Visual grounding coordinates from /parse API (only for chunks derived from
    embedded images)
    """


class DataSpreadsheetParseResponseMetadata(BaseModel):
    """Metadata for spreadsheet parsing result."""

    duration_ms: int
    """Processing duration in milliseconds"""

    filename: str
    """Original filename"""

    sheet_count: int
    """Number of sheets processed"""

    total_cells: int
    """Total non-empty cells across all sheets"""

    total_chunks: int
    """Total chunks (tables + images) extracted"""

    total_rows: int
    """Total rows across all sheets"""

    credit_usage: Optional[float] = None
    """Credits charged"""

    job_id: Optional[str] = None
    """Inference history job ID"""

    org_id: Optional[str] = None
    """Organization ID"""

    total_images: Optional[int] = None
    """Total images extracted"""

    version: Optional[str] = None
    """Model version for parsing images"""


class DataSpreadsheetParseResponseSplit(BaseModel):
    """Sheet-based split from spreadsheet parsing.

    Similar to ParseSplit but grouped by sheet instead of page.
    Supports both 'page' (per-sheet) and 'full' (all sheets) split types.
    """

    chunks: List[str]
    """Chunk IDs in this split"""

    class_: str = FieldInfo(alias="class")
    """
    Split class: 'page' for per-sheet splits, 'full' for single split with all
    content
    """

    identifier: str
    """Split identifier: sheet name for 'page' splits, 'full' for full split"""

    markdown: str
    """Combined markdown for this split"""

    sheets: List[int]
    """Sheet indices: single element for 'page' splits, all indices for 'full' split"""


class DataSpreadsheetParseResponse(BaseModel):
    """Response from /ade/parse-spreadsheet endpoint.

    Similar structure to ParseResponse but without grounding.
    """

    chunks: List[DataSpreadsheetParseResponseChunk]
    """List of table chunks (HTML)"""

    markdown: str
    """Full document as HTML with anchor tags and tables"""

    metadata: DataSpreadsheetParseResponseMetadata
    """Metadata for spreadsheet parsing result."""

    splits: List[DataSpreadsheetParseResponseSplit]
    """Sheet-based splits"""


Data: TypeAlias = Union[DataParseResponse, DataSpreadsheetParseResponse, None]


class ParseJobGetResponse(BaseModel):
    """Unified response for job status endpoint."""

    job_id: str

    progress: float
    """
    Job completion progress as a decimal from 0 to 1, where 0 is not started, 1 is
    finished, and values between 0 and 1 indicate work in progress.
    """

    received_at: int

    status: str

    data: Optional[Data] = None
    """
    The parsed output (ParseResponse for documents, SpreadsheetParseResponse for
    spreadsheets), if the job is complete and the `output_save_url` parameter was
    not used.
    """

    failure_reason: Optional[str] = None

    metadata: Optional[ParseMetadata] = None

    org_id: Optional[str] = None

    output_url: Optional[str] = None
    """The URL to the parsed content.

    This field contains a URL when the job is complete and either you specified the
    `output_save_url` parameter or the result is larger than 1MB. When the result
    exceeds 1MB, the URL is a presigned S3 URL that expires after 1 hour. Each time
    you GET the job, a new presigned URL is generated.
    """

    version: Optional[str] = None
