# src/landingai_ade/resources/v2/files.py
from __future__ import annotations

from typing import Mapping, cast

import httpx

from ._base import V2ResourceMixin
from ..._files import deepcopy_with_paths
from ..._types import Body, Query, Headers, NotGiven, FileTypes, not_given
from ..._utils import extract_files
from ...types.v2 import V2FileUploadResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._exceptions import LandingAiadeError
from ..._base_client import make_request_options

__all__ = ["FilesResource", "AsyncFilesResource"]


class FilesResource(V2ResourceMixin, SyncAPIResource):
    def upload(
        self,
        *,
        file: FileTypes,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> str:
        """Stage bytes on the data plane; returns a `file_ref` for use in job inputs."""
        body = deepcopy_with_paths({"file": file}, [["file"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["file"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        response = self._post(
            self._v2_url("/v1/files"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2FileUploadResponse,
        )
        if not response.file_ref:
            raise LandingAiadeError(f"POST /v1/files did not return a file_ref (got: {response!r}).")
        return response.file_ref


class AsyncFilesResource(V2ResourceMixin, AsyncAPIResource):
    async def upload(
        self,
        *,
        file: FileTypes,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> str:
        """Stage bytes on the data plane; returns a `file_ref` for use in job inputs."""
        body = deepcopy_with_paths({"file": file}, [["file"]])
        files = extract_files(cast(Mapping[str, object], body), paths=[["file"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        response = await self._post(
            self._v2_url("/v1/files"),
            body=body,
            files=files,
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=V2FileUploadResponse,
        )
        if not response.file_ref:
            raise LandingAiadeError(f"POST /v1/files did not return a file_ref (got: {response!r}).")
        return response.file_ref
