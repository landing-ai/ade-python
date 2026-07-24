# src/landingai_ade/resources/v2/v2.py
from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union, Mapping, Optional, Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel

from ._base import V2ResourceMixin
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._compat import cached_property
from ...types.v2 import V2GroundResult, V2ExtractResult, V2ParseResponse, V2BuildSchemaResponse
from ..._resource import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from .files import FilesResource, AsyncFilesResource
    from .parse import ParseResource, ParseJobsResource, AsyncParseResource, AsyncParseJobsResource
    from .ground import GroundResource, GroundJobsResource, AsyncGroundResource, AsyncGroundJobsResource
    from .extract import ExtractResource, ExtractJobsResource, AsyncExtractResource, AsyncExtractJobsResource
    from .build_schema import (
        BuildSchemaResource,
        BuildSchemaJobsResource,
        AsyncBuildSchemaResource,
        AsyncBuildSchemaJobsResource,
    )

__all__ = ["V2Resource", "AsyncV2Resource"]


class V2Resource(SyncAPIResource, V2ResourceMixin):
    """Container for the V2 (ADE) surface: ``client.v2.<resource>``.

    ``files``, ``parse``, ``extract``, and ``ground`` are wired up; each sub-resource does its
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

    @cached_property
    def extract_jobs(self) -> ExtractJobsResource:
        from .extract import ExtractJobsResource

        return ExtractJobsResource(self._client)

    def extract(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
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
            markdown_url=markdown_url,
            model=model,
            strict=strict,
            save_to=save_to,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

    @cached_property
    def _build_schema(self) -> BuildSchemaResource:
        from .build_schema import BuildSchemaResource

        return BuildSchemaResource(self._client)

    @cached_property
    def build_schema_jobs(self) -> BuildSchemaJobsResource:
        from .build_schema import BuildSchemaJobsResource

        return BuildSchemaJobsResource(self._client)

    def build_schema(
        self,
        *,
        markdowns: Optional[Sequence[FileTypes]] | Omit = omit,
        markdown_urls: Optional[Sequence[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Union[str, Mapping[str, object], Type[BaseModel], None] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2BuildSchemaResponse:
        """Generate or edit a JSON Schema for extraction synchronously. See ``BuildSchemaResource.run`` for full documentation."""
        return self._build_schema.run(
            markdowns=markdowns,
            markdown_urls=markdown_urls,
            prompt=prompt,
            schema=schema,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

    @cached_property
    def _ground(self) -> GroundResource:
        from .ground import GroundResource

        return GroundResource(self._client)

    @cached_property
    def ground_jobs(self) -> GroundJobsResource:
        from .ground import GroundJobsResource

        return GroundJobsResource(self._client)

    def ground(
        self,
        *,
        extraction_metadata: Union[Mapping[str, object], BaseModel],
        structure: Union[Mapping[str, object], BaseModel],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2GroundResult:
        """Ground extracted fields to document blocks synchronously. See ``GroundResource.run`` for full documentation."""
        return self._ground.run(
            extraction_metadata=extraction_metadata,
            structure=structure,
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

    @cached_property
    def extract_jobs(self) -> AsyncExtractJobsResource:
        from .extract import AsyncExtractJobsResource

        return AsyncExtractJobsResource(self._client)

    async def extract(
        self,
        *,
        schema: Union[str, Mapping[str, object], Type[BaseModel]],
        markdown: Optional[str] | Omit = omit,
        markdown_url: Optional[str] | Omit = omit,
        model: Optional[str] | Omit = omit,
        strict: Optional[bool] | Omit = omit,
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
            markdown_url=markdown_url,
            model=model,
            strict=strict,
            save_to=save_to,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

    @cached_property
    def _build_schema(self) -> AsyncBuildSchemaResource:
        from .build_schema import AsyncBuildSchemaResource

        return AsyncBuildSchemaResource(self._client)

    @cached_property
    def build_schema_jobs(self) -> AsyncBuildSchemaJobsResource:
        from .build_schema import AsyncBuildSchemaJobsResource

        return AsyncBuildSchemaJobsResource(self._client)

    async def build_schema(
        self,
        *,
        markdowns: Optional[Sequence[FileTypes]] | Omit = omit,
        markdown_urls: Optional[Sequence[str]] | Omit = omit,
        prompt: Optional[str] | Omit = omit,
        schema: Union[str, Mapping[str, object], Type[BaseModel], None] | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2BuildSchemaResponse:
        """Async mirror of :meth:`V2Resource.build_schema`."""
        return await self._build_schema.run(
            markdowns=markdowns,
            markdown_urls=markdown_urls,
            prompt=prompt,
            schema=schema,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

    @cached_property
    def _ground(self) -> AsyncGroundResource:
        from .ground import AsyncGroundResource

        return AsyncGroundResource(self._client)

    @cached_property
    def ground_jobs(self) -> AsyncGroundJobsResource:
        from .ground import AsyncGroundJobsResource

        return AsyncGroundJobsResource(self._client)

    async def ground(
        self,
        *,
        extraction_metadata: Union[Mapping[str, object], BaseModel],
        structure: Union[Mapping[str, object], BaseModel],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> V2GroundResult:
        """Async mirror of :meth:`V2Resource.ground`."""
        return await self._ground.run(
            extraction_metadata=extraction_metadata,
            structure=structure,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
