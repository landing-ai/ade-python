# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Optional
from typing_extensions import Literal

from pydantic import Field as FieldInfo

from .._models import BaseModel
from .shared.parse_metadata import ParseMetadata
from .shared.parse_grounding_box import ParseGroundingBox

__all__ = ["ParseJobGetResponse", "Data", "DataChunk", "DataChunkGrounding", "DataSplit", "DataGrounding"]


class DataChunkGrounding(BaseModel):
    box: ParseGroundingBox

    page: int


class DataChunk(BaseModel):
    id: str

    grounding: DataChunkGrounding

    markdown: str

    type: str


class DataSplit(BaseModel):
    chunks: List[str]

    class_: str = FieldInfo(alias="class")

    identifier: str

    markdown: str

    pages: List[int]


class DataGrounding(BaseModel):
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


class Data(BaseModel):
    chunks: List[DataChunk]

    markdown: str

    metadata: ParseMetadata

    splits: List[DataSplit]

    grounding: Optional[Dict[str, DataGrounding]] = None


class ParseJobGetResponse(BaseModel):
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
    The parsed output, if the job is complete and the `output_save_url` parameter
    was not used.
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
