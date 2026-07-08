# src/landingai_ade/resources/v2/v2.py
from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union, Mapping, Optional
from pathlib import Path

import httpx
from pydantic import BaseModel

from ._base import V2ResourceMixin
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._compat import cached_property
from ...types.v2 import V2ExtractResult, V2ParseResponse
from ..._resource import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from .files import FilesResource, AsyncFilesResource
    from .parse import ParseResource, ParseJobsResource, AsyncParseResource, AsyncParseJobsResource
    from .extract import ExtractResource, AsyncExtractResource

__all__ = ["V2Resource", "AsyncV2Resource"]


class V2Resource(SyncAPIResource, V2ResourceMixin):
    """Container for the V2 (ADE) surface: ``client.v2.<resource>``.

    ``files``, ``parse``, and ``extract`` are wired up; each sub-resource does its
    own lazy import inside its cached property body -- mirroring
    ``LandingAIADE.parse_jobs`` -- so that this module keeps importing standalone
    regardless of which sub-resources exist yet. Remaining job-polling resources
    are attached by later tasks following the same pattern.
    """

    @cached_property
    def files(self) -> FilesResource:
        from .files import FilesResource

        return FilesResource(self._client)

    @cached_property
    def _parse(self) -> ParseResource:
        from .parse import ParseResource

        return ParseResource(self._client)

    @cached_property
    def parse_jobs(self) -> ParseJobsResource:
        from .parse import ParseJobsResource

        return ParseJobsResource(self._client)

    def parse(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ParseResponse:
        """Parse a document synchronously. See ``ParseResource.run`` for full documentation."""
        return self._parse.run(
            document=document,
            document_url=document_url,
            model=model,
            options=options,
            password=password,
            save_to=save_to,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

    @cached_property
    def _extract(self) -> ExtractResource:
        from .extract import ExtractResource

        return ExtractResource(self._client)

    def extract(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        """Extract structured data from markdown synchronously. See ``ExtractResource.run`` for full documentation."""
        return self._extract.run(
            schema=schema,
            markdown=markdown,
            markdown_ref=markdown_ref,
            markdown_url=markdown_url,
            model=model,
            strict=strict,
            idempotency_key=idempotency_key,
            save_to=save_to,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )


class AsyncV2Resource(AsyncAPIResource, V2ResourceMixin):
    """Async mirror of :class:`V2Resource`."""

    @cached_property
    def files(self) -> AsyncFilesResource:
        from .files import AsyncFilesResource

        return AsyncFilesResource(self._client)

    @cached_property
    def _parse(self) -> AsyncParseResource:
        from .parse import AsyncParseResource

        return AsyncParseResource(self._client)

    @cached_property
    def parse_jobs(self) -> AsyncParseJobsResource:
        from .parse import AsyncParseJobsResource

        return AsyncParseJobsResource(self._client)

    async def parse(
        self,
        *,
        document: Optional[FileTypes] | Omit = omit,
        document_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        options: Optional[Mapping[str, object]] | Omit = omit,
        password: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ParseResponse:
        """Async mirror of :meth:`V2Resource.parse`."""
        return await self._parse.run(
            document=document,
            document_url=document_url,
            model=model,
            options=options,
            password=password,
            save_to=save_to,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

    @cached_property
    def _extract(self) -> AsyncExtractResource:
        from .extract import AsyncExtractResource

        return AsyncExtractResource(self._client)

    async def extract(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_ref: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
        idempotency_key: Optional[str] | Omit = omit,
        save_to: str | Path | None = None,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2ExtractResult:
        """Async mirror of :meth:`V2Resource.extract`."""
        return await self._extract.run(
            schema=schema,
            markdown=markdown,
            markdown_ref=markdown_ref,
            markdown_url=markdown_url,
            model=model,
            strict=strict,
            idempotency_key=idempotency_key,
            save_to=save_to,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
