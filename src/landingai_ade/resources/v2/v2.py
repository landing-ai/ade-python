# src/landingai_ade/resources/v2/v2.py
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional
from pathlib import Path

import httpx

from ._base import V2ResourceMixin
from ..._types import Body, Omit, Query, Headers, NotGiven, FileTypes, omit, not_given
from ..._compat import cached_property
from ...types.v2 import V2ParseResponse
from ..._resource import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from .files import FilesResource, AsyncFilesResource
    from .parse import ParseResource, AsyncParseResource

__all__ = ["V2Resource", "AsyncV2Resource"]


class V2Resource(SyncAPIResource, V2ResourceMixin):
    """Container for the V2 (ADE) surface: ``client.v2.<resource>``.

    ``files`` and ``parse`` are wired up; each sub-resource does its own lazy
    import inside its cached property body -- mirroring ``LandingAIADE.parse_jobs``
    -- so that this module keeps importing standalone regardless of which
    sub-resources exist yet. ``extract`` and the job-polling resources are
    attached by later tasks following the same pattern.
    """

    @cached_property
    def files(self) -> FilesResource:
        from .files import FilesResource

        return FilesResource(self._client)

    @cached_property
    def _parse(self) -> ParseResource:
        from .parse import ParseResource

        return ParseResource(self._client)

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
